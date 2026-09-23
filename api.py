"""
Minimal interface for using the Si I GraphNet NLTE model from an external inversion code.

Three functions, all operating on ONE 1D atmospheric column at a time:

    compute_dep_coeffs(T, z, ne, vturb, vlos)
        GraphNet inference only, no lightweaver dependency. One column costs ~170 ms on CPU
        and ~50 ms on a GPU (600 layers on 211 nodes is launch-bound, not compute-bound).

    compute_dep_coeffs_batch([(T, z, ne, vturb, vlos), ...])
        Same, for many columns in one forward pass. On a GPU 64 columns take ~80 ms, i.e.
        ~1.3 ms per column -- this is the function to call inside the hot loop of an inversion.

    synthesis_lw(T, z, ne, vturb, vlos)
        Full lightweaver NLTE solve (iterates statistical equilibrium to convergence),
        exactly as used to build the training set. Slow -- use as ground truth / for
        validating the GraphNet, not inside an inversion loop.

    intensity_gnn(T, z, ne, vturb, vlos)
        NLTE-accelerated synthesis: GraphNet for the departure coefficients + two lightweaver
        formal solutions (one to obtain the mean radiation field for the background scattering
        term, one for the emergent rays), with no statistical-equilibrium iteration. Same
        (wave, Iwave, log_dep) return signature as `synthesis_lw`, so it is a drop-in, much
        faster replacement.

Units (SI, matching how the training set was generated)
---------------------------------------------------------
    T      temperature             [K]
    z      geometric height        [m]     (increasing upward, z=0 near the photosphere)
    ne     electron density        [m^-3]
    vturb  microturbulent velocity [m/s]
    vlos   line-of-sight velocity  [m/s]

Usage
-----
    import sys
    sys.path.insert(0, '/path/to/graphnet_nlte')
    import api

    log_dep = api.compute_dep_coeffs(T, z, ne, vturb, vlos)
    log_deps = api.compute_dep_coeffs_batch([(T, z, ne, vturb, vlos), ...], device='cuda:0')
    wave, Iwave, log_dep = api.intensity_gnn(T, z, ne, vturb, vlos)

Dependencies
------------
    compute_dep_coeffs   : numpy, torch, torch_geometric
    synthesis_lw/intensity_gnn : the above, plus lightweaver (imported lazily, only
                                 when one of these two functions is actually called)
"""
import glob
import os
import sys
import warnings

import numpy as np
import torch

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import graphnet

# Default model, chosen when this module is imported:
#
#   * a file 'checkpoint.pth' next to api.py, if there is one. This is the layout of the
#     standalone distribution (api_standalone/), which ships one pinned model;
#   * otherwise the checkpoint tree 'checkpoints_si_v3' next to api.py, i.e. the development
#     layout. That is a directory, not a file: Formal.py overwrites '<run>/best.pth' each time
#     the validation loss improves, so the last '*best.pth' (sorted by path) anywhere below it is
#     the best checkpoint of the most recent run. It is resolved at call time (see
#     _resolve_checkpoint), so it keeps tracking training without being edited by hand.
#
# Pass checkpoint= explicitly to pin a specific model in either layout.
_STANDALONE_CHECKPOINT = os.path.join(_MODULE_DIR, 'checkpoint.pth')
DEFAULT_CHECKPOINT = (_STANDALONE_CHECKPOINT if os.path.isfile(_STANDALONE_CHECKPOINT)
                      else os.path.join(_MODULE_DIR, 'checkpoints_si_v3'))

# Si I 10827 A window (the line sits at 1083.0038 nm in vacuum). The sampling has to resolve the
# Doppler width, which for Si at the coolest temperature in the training set (2500 K, vturb=0)
# is 1083 nm * sqrt(2kT/m)/c = 4.4 pm; 2 pm gives at least two samples across it everywhere. An
# earlier 1100-point version of this grid sampled at 10 pm -- coarser than the Doppler width --
# and read the line core 0.2-4% too shallow.
DEFAULT_WAVE = np.linspace(1074.0, 1085.0, 5501)

_MODEL_CACHE = {}


class AtmosphereError(ValueError):
    """Raised when the input atmospheric arrays are missing, inconsistent, or unphysical."""


def _validate_atmosphere(T, z, ne, vturb, vlos):
    """Coerce inputs to 1D float64 arrays and check they describe a valid atmosphere."""
    names = ('T', 'z', 'ne', 'vturb', 'vlos')
    raw = (T, z, ne, vturb, vlos)

    arrays = []
    for name, val in zip(names, raw):
        if val is None:
            raise AtmosphereError(f"'{name}' is required and cannot be None")
        # ascontiguousarray (not asarray): lightweaver's Cython layer requires C-contiguous
        # memory and fails with an opaque "ndarray is not C-contiguous" error deep inside
        # its own internals otherwise -- easy to hit by passing a reversed/sliced view (e.g.
        # arr[::-1] to flip height ordering), so guard against it once here for all callers.
        arr = np.ascontiguousarray(val, dtype=np.float64)
        if arr.ndim != 1:
            raise AtmosphereError(f"'{name}' must be a 1D array, got shape {arr.shape}")
        arrays.append(arr)
    T, z, ne, vturb, vlos = arrays

    lengths = {name: len(a) for name, a in zip(names, arrays)}
    if len(set(lengths.values())) > 1:
        raise AtmosphereError(f"All atmospheric arrays must have the same length, got {lengths}")
    if len(T) < 2:
        raise AtmosphereError(f"Atmosphere must have at least 2 depth points, got {len(T)}")

    for name, arr in zip(names, arrays):
        if not np.all(np.isfinite(arr)):
            raise AtmosphereError(f"'{name}' contains NaN or Inf values")

    if np.any(T <= 0):
        raise AtmosphereError("'T' must be strictly positive (Kelvin)")
    if np.any(ne <= 0):
        raise AtmosphereError("'ne' must be strictly positive (m^-3)")
    if np.any(vturb < 0):
        raise AtmosphereError("'vturb' cannot be negative (m/s)")
    if not np.all(np.diff(z) < 0):
        raise AtmosphereError(
            "'z' must be strictly decreasing (index 0 = top of the atmosphere, highest z, "
            "down to the last index = deepest point) -- this matches lightweaver's geometric "
            "scale convention and how the training set was generated (e.g. Bifrost columns "
            "are reversed with z[::-1] before use). Flip all five arrays with [::-1] if your "
            "atmosphere is ordered the other way."
        )

    return T, z, ne, vturb, vlos


def _build_graph(T, z, ne, vturb, vlos, norm_stats):
    """
    Build the normalized node/edge tensors for one atmosphere column. Mirrors the feature
    construction in Dataset.py (same chain-graph connectivity) for the node_input_size=5 /
    edge_input_size=1 configuration, without any pickle/KDTree overhead. `norm_stats` must
    be the exact NORM_STATS dict the target checkpoint was trained with (see _load_model).
    """
    n = len(T)

    node = np.empty((n, 5), dtype=np.float32)
    node[:, 0] = (np.log10(T) - norm_stats['T_log10']['mean']) / norm_stats['T_log10']['std']
    node[:, 1] = (z - norm_stats['z']['mean']) / norm_stats['z']['std']
    node[:, 2] = (np.log10(ne) - norm_stats['ne_log10']['mean']) / norm_stats['ne_log10']['std']
    node[:, 3] = (vturb / 1e3 - norm_stats['vturb_km']['mean']) / norm_stats['vturb_km']['std']
    node[:, 4] = (vlos / 1e3 - norm_stats['vlos_km']['mean']) / norm_stats['vlos_km']['std']

    # Chain graph: node i <-> i-1 and i <-> i+1 (equivalent to Dataset.py's radius=1 KDTree
    # query on a 1D index array, without the KDTree).
    senders = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    receivers = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    edge_index = np.stack([senders, receivers]).astype(np.int64)

    edge_attr = ((z[senders] - z[receivers]) / norm_stats['delta_z']['std']).astype(np.float32).reshape(-1, 1)

    return torch.from_numpy(node), torch.from_numpy(edge_index), torch.from_numpy(edge_attr)


def _resolve_checkpoint(checkpoint):
    """
    If `checkpoint` is a directory, resolve it to the most recent '*best.pth' file at or below
    it. Formal.py overwrites '<run>/best.pth' on each improvement and run directories are named
    by timestamp, so the lexicographically-last path is the best checkpoint of the latest run
    (older trees with one timestamped '<stamp>_best.pth' per improvement are matched too). The
    search is recursive so that either a single run directory or the whole checkpoint tree can
    be given. If `checkpoint` is already a file, it is returned unchanged. Directories are
    re-resolved on every call, so a still-training run always yields its current best checkpoint.
    """
    if os.path.isdir(checkpoint):
        candidates = sorted(glob.glob(os.path.join(checkpoint, '**', '*best.pth'), recursive=True))
        if not candidates:
            raise FileNotFoundError(f"No '*best.pth' checkpoint found under directory: {checkpoint}")
        return candidates[-1]
    return checkpoint


def _load_model(checkpoint, device):
    """
    Load (and cache) a GraphNet checkpoint, along with the exact normalization constants it
    was trained with. Repeated calls are free after the first (per resolved checkpoint file).
    """
    checkpoint = _resolve_checkpoint(checkpoint)
    key = (os.path.abspath(checkpoint), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"GraphNet checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    hyperparams = dict(ckpt['hyperparameters'])
    if hyperparams.get('node_input_size') != 5 or hyperparams.get('edge_input_size') != 1:
        raise NotImplementedError(
            "api.py only supports checkpoints trained with node_input_size=5 (T, z, ne, "
            f"vturb, vlos) and edge_input_size=1 (delta z); got node_input_size="
            f"{hyperparams.get('node_input_size')}, edge_input_size={hyperparams.get('edge_input_size')}."
        )
    norm_stats = ckpt.get('norm_stats')
    if norm_stats is None:
        raise ValueError(
            f"Checkpoint {checkpoint} has no embedded 'norm_stats' -- it predates that being "
            "saved by Formal.py and is not supported. Use a checkpoint trained with the "
            "current Formal.py/Dataset.py."
        )

    model = graphnet.EncodeProcessDecode(**hyperparams).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    _MODEL_CACHE[key] = (model, hyperparams, norm_stats)
    return _MODEL_CACHE[key]


def compute_dep_coeffs(T, z, ne, vturb, vlos, checkpoint=DEFAULT_CHECKPOINT, device='cpu'):
    """
    Predict NLTE departure coefficients for one atmospheric column with the GraphNet model.

    Pure GNN inference (no lightweaver call). Measured cost per call: ~170 ms on CPU and
    ~50 ms on an H100 -- the network is 600 layers deep and one column is far too small to fill
    a GPU, so the time is kernel-launch overhead. For many columns use `compute_dep_coeffs_batch`
    instead (~1.3 ms per column on a GPU). The model is cached after the first call, so repeated
    calls with the same `checkpoint` only pay the load cost once per process.

    Parameters
    ----------
    T, z, ne, vturb, vlos : array_like, shape (n_depth,)
        Atmospheric stratification, see module docstring for units. `ne` is required: unlike
        a full lightweaver solve, the network cannot derive it on its own.
    checkpoint : str, optional
        Path to a trained GraphNet checkpoint (*_best.pth), or a directory containing one or
        more -- in which case the most recent '*_best.pth' in it is used (re-resolved on every
        call, so pointing this at an in-progress training run's directory always picks up its
        current best checkpoint). Selects which training run to use.
    device : str, optional
        torch device for inference, e.g. 'cpu' or 'cuda:0'.

    Returns
    -------
    log_dep : ndarray, shape (n_levels, n_depth)
        log10(b) = log10(n / n*) for every atomic level and depth point, in the same
        (n_levels, n_depth) layout as lightweaver's `ctx.activeAtoms[0].n`.

    Raises
    ------
    AtmosphereError
        Input arrays are missing, inconsistent, too short, non-finite, or unphysical.
    FileNotFoundError
        `checkpoint` does not exist, or is a directory with no '*_best.pth' file in it.
    NotImplementedError
        `checkpoint` was trained with an unsupported node/edge feature configuration.
    ValueError
        `checkpoint` has no embedded normalization constants (too old / incompatible).
    """
    return compute_dep_coeffs_batch([(T, z, ne, vturb, vlos)], checkpoint=checkpoint, device=device)[0]


def compute_dep_coeffs_batch(columns, checkpoint=DEFAULT_CHECKPOINT, device='cpu'):
    """
    Predict NLTE departure coefficients for many atmospheric columns in one forward pass.

    The columns are concatenated into a single disconnected graph (the same batching the
    training loop uses), so the cost is one network evaluation regardless of how many columns
    are passed. On an H100, 64 columns of 211 points take ~80 ms (~1.3 ms per column) against
    ~50 ms for a single column; on CPU the gain is smaller but still worthwhile. Columns may
    have different numbers of depth points.

    Parameters
    ----------
    columns : sequence of (T, z, ne, vturb, vlos) tuples
        One tuple per column, each element an array of shape (n_depth,); see the module
        docstring for units and ordering. Every column is validated as in `compute_dep_coeffs`.
    checkpoint, device : optional
        See `compute_dep_coeffs`.

    Returns
    -------
    log_deps : list of ndarray, each of shape (n_levels, n_depth)
        log10(b) for each column, in the order given.

    Raises
    ------
    Same as `compute_dep_coeffs`; an invalid column raises before anything is evaluated.
    """
    columns = list(columns)
    if not columns:
        return []
    model, _, norm_stats = _load_model(checkpoint, device)

    nodes, edge_indices, edge_attrs, batch_vec, lengths = [], [], [], [], []
    offset = 0
    for g, col in enumerate(columns):
        T, z, ne, vturb, vlos = _validate_atmosphere(*col)
        node, edge_index, edge_attr = _build_graph(T, z, ne, vturb, vlos, norm_stats)
        nodes.append(node)
        edge_indices.append(edge_index + offset)
        edge_attrs.append(edge_attr)
        batch_vec.append(torch.full((len(T),), g, dtype=torch.long))
        lengths.append(len(T))
        offset += len(T)

    node = torch.cat(nodes).to(device)
    edge_index = torch.cat(edge_indices, dim=1).to(device)
    edge_attr = torch.cat(edge_attrs).to(device)
    batch = torch.cat(batch_vec).to(device)
    u = torch.zeros((len(columns), 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(node, edge_attr, edge_index, u, batch)

    out = out.cpu().numpy() * 5.0  # (sum n_depth, n_levels)

    if not np.all(np.isfinite(out)):
        raise RuntimeError("GraphNet produced non-finite departure coefficients")

    # Clamp to the range the targets were clipped to. Inside it this is a no-op; outside it
    # guards the levels and depths that carry n_i/n_Total < 1e-9 and are therefore masked out of
    # the training loss (see Dataset.NEGLIGIBLE_LOG_N_OVER_NTOT) -- the network is unsupervised
    # there and its raw output can be arbitrary, while 10**log_dep feeds straight into the
    # populations. At +-10 the resulting populations are bounded by what the clipped targets
    # themselves produce, which changes the emergent profile by <= 2e-7.
    log_deps, start = [], 0
    for n in lengths:
        log_deps.append(np.clip(out[start:start + n].T, -10.0, 10.0))  # -> (n_levels, n_depth)
        start += n
    return log_deps


# Below this wavelength intensity_gnn is not accurate: its mean radiation field comes from a
# single formal solution, which is enough where scattering is a small perturbation but not in
# the ultraviolet, where the background scattering emissivity and the Si I bound-free edges make
# J far from its converged value. Measured against the converged solve (|dI/I| median / max):
# 6e-8 / 8e-8 in the 1083 nm window, 7e-7 / 2e-5 at 400-800 nm, 2e-4 / 2e-2 at 200-400 nm and
# 4e-4 / 0.8 at 100-200 nm.
_SCATTERING_SAFE_MIN_NM = 400.0


def _warn_if_scattering_matters(wave):
    """Warn when intensity_gnn is asked for wavelengths where its J = 0 assumption breaks."""
    wave = np.asarray(wave)
    if wave.size and wave.min() < _SCATTERING_SAFE_MIN_NM:
        warnings.warn(
            f"intensity_gnn obtains the mean radiation field from a single formal solution "
            f"instead of iterating it. This is accurate above {_SCATTERING_SAFE_MIN_NM:.0f} nm "
            f"(|dI/I| below 2e-5) but not in the ultraviolet (up to ~2e-2 at 200-400 nm and "
            f"~0.8 at 100-200 nm), and the requested grid reaches {wave.min():.1f} nm. Use "
            f"synthesis_lw for those wavelengths.",
            RuntimeWarning, stacklevel=3)


def _build_atmosphere(T, z, ne, vturb, vlos):
    """Shared lightweaver atmosphere/population setup for synthesis_lw and intensity_gnn."""
    import lightweaver as lw
    from lightweaver.rh_atoms import (H_6_atom, C_atom, OI_ord_atom, Al_atom,
                                       CaII_atom, Fe_atom, He_9_atom, MgII_atom, N_atom, Na_atom, S_atom)
    # Si_atom_custom is not in released lightweaver -- it ships with this repo (si_atom.py).
    from si_atom import Si_atom_custom

    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=z, temperature=T,
                                   vlos=vlos, vturb=vturb, ne=ne, verbose=False)
    atmos.quadrature(5)

    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom_custom(), Al_atom(),
                            CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('Si')
    spect = aSet.compute_wavelength_grid()
    eqPops = aSet.compute_eq_pops(atmos)

    return atmos, aSet, spect, eqPops


def synthesis_lw(T, z, ne, vturb, vlos, wave=None, mu=None, conserve_charge=False, prd=False):
    """
    Reference NLTE synthesis with a full lightweaver solve, exactly as used to build the
    training set (see generate_database.py). Iterates statistical equilibrium to
    convergence -- orders of magnitude slower than `intensity_gnn`. Use as ground truth
    or to validate the GraphNet, not inside an inversion's hot loop.

    Parameters
    ----------
    T, z, ne, vturb, vlos : array_like, shape (n_depth,)
        See module docstring for units.
    wave : array_like, optional
        Wavelength grid [nm]. Defaults to the Si I 10827 A window used for training
        (1074-1085 nm, 1100 points).
    mu : float, optional
        Cosine of the viewing angle. Defaults to atmos.muz[-1], the outermost node of the
        5-point Gauss-Legendre quadrature -- mu = 0.9531, i.e. theta = 17.6 deg, not disk
        centre. This matches training-set generation; pass mu=1.0 for disk centre.
    conserve_charge : bool, optional
        Solve for charge-conserving electron density during the NLTE iteration.
        False (default) matches training-set generation.
    prd : bool, optional
        Use partial redistribution. False (default) matches training-set generation.

    Returns
    -------
    wave : ndarray, shape (n_wave,)
    Iwave : ndarray, shape (n_wave,)
        Emergent intensity.
    log_dep : ndarray, shape (n_levels, n_depth)
        Converged log10(n / n*) departure coefficients.

    Raises
    ------
    AtmosphereError
        Input arrays are missing, inconsistent, too short, non-finite, or unphysical.
    RuntimeError
        The NLTE iteration failed to converge (non-finite populations).
    """
    import lightweaver as lw

    T, z, ne, vturb, vlos = _validate_atmosphere(T, z, ne, vturb, vlos)
    if wave is None:
        wave = DEFAULT_WAVE

    atmos, aSet, spect, eqPops = _build_atmosphere(T, z, ne, vturb, vlos)
    if mu is None:
        mu = atmos.muz[-1]

    ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=conserve_charge)
    lw.iterate_ctx_se(ctx, prd=prd, quiet=True)
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    ctx.formal_sol_gamma_matrices()
    if prd:
        ctx.prd_redistribute()

    log_dep = np.log10(ctx.activeAtoms[0].n / ctx.activeAtoms[0].nStar)
    if not np.all(np.isfinite(log_dep)):
        raise RuntimeError("lightweaver NLTE iteration did not converge (non-finite departure coefficients)")

    Iwave = ctx.compute_rays(wave, [mu], stokes=False)

    return np.asarray(wave), np.asarray(Iwave), log_dep


def intensity_gnn(T, z, ne, vturb, vlos, log_dep=None, wave=None, mu=None,
                   checkpoint=DEFAULT_CHECKPOINT, device='cpu'):
    """
    NLTE-accelerated synthesis: GraphNet for the departure coefficients (unless already
    supplied) + one lightweaver formal solution for the mean radiation field + the emergent
    rays, with no statistical-equilibrium iteration. Returns the same (wave, Iwave, log_dep)
    triple as `synthesis_lw`, so the two are interchangeable -- this is the fast function to
    use inside an inversion.

    Parameters
    ----------
    T, z, ne, vturb, vlos : array_like, shape (n_depth,)
        See module docstring for units.
    log_dep : ndarray, shape (n_levels, n_depth), optional
        Precomputed departure coefficients (e.g. from a prior `compute_dep_coeffs` call),
        to avoid recomputing them. If None (default), they are predicted here.
    wave, mu : optional
        See `synthesis_lw`. `mu` defaults to 0.9531 (theta = 17.6 deg), not disk centre.
    checkpoint, device : optional
        See `compute_dep_coeffs`. Ignored if `log_dep` is supplied directly.

    Returns
    -------
    wave : ndarray, shape (n_wave,)
    Iwave : ndarray, shape (n_wave,)
        Emergent intensity.
    log_dep : ndarray, shape (n_levels, n_depth)
        The departure coefficients used (as supplied, or as predicted by the GraphNet).

    Raises
    ------
    AtmosphereError
        Input arrays are missing, inconsistent, too short, non-finite, or unphysical.
    FileNotFoundError, NotImplementedError, ValueError, RuntimeError
        Only if `log_dep` is not supplied: propagated from the internal
        `compute_dep_coeffs` call, see there.
    """
    T, z, ne, vturb, vlos = _validate_atmosphere(T, z, ne, vturb, vlos)
    if wave is None:
        wave = DEFAULT_WAVE
    _warn_if_scattering_matters(wave)
    if log_dep is None:
        log_dep = compute_dep_coeffs(T, z, ne, vturb, vlos, checkpoint=checkpoint, device=device)

    atmos, aSet, spect, eqPops = _build_atmosphere(T, z, ne, vturb, vlos)
    if mu is None:
        mu = atmos.muz[-1]
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)

    nstar = eqPops.atomicPops['Si'].nStar
    eqPops.atomicPops['Si'].n = (10.0 ** log_dep) * nstar

    import lightweaver as lw
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
    # One formal solution on the full grid before the rays: a fresh Context has J = 0, so the
    # background coherent-scattering emissivity (sigma J) would be missing from the source
    # function. This pass fills J from the populations just set and leaves them untouched.
    # Measured against the converged solve in the Si I 1083 nm window: the RMS error of the
    # emergent profile drops from ~2e-4 to ~5e-8 of the continuum, for 10-25 ms per column.
    ctx.formal_sol_gamma_matrices()
    Iwave = ctx.compute_rays(wave, [mu], stokes=False)

    return np.asarray(wave), np.asarray(Iwave), log_dep


if __name__ == '__main__':
    # Lightweight smoke test for compute_dep_coeffs (no lightweaver needed). Uses a
    # simple isothermal-ish toy column purely to exercise the code path end-to-end.
    n = 60
    z = np.linspace(2.5e6, -5e5, n)                      # m, decreasing: index 0 = top
    T = 6000.0 + 4000.0 * np.exp(-((z - 1e6) / 8e5)**2)  # K
    ne = 10.0 ** np.linspace(17.0, 21.0, n)              # m^-3, rising with depth
    vturb = np.full(n, 2.0e3)                            # m/s
    vlos = np.zeros(n)                                   # m/s

    log_dep = compute_dep_coeffs(T, z, ne, vturb, vlos)
    print('log_dep shape:', log_dep.shape)
    print('log_dep range:', log_dep.min(), log_dep.max())
    print('compute_dep_coeffs smoke test passed')
