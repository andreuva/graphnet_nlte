"""
Minimal interface for using the Si I GraphNet NLTE model from an external inversion code.

Three functions, all operating on ONE 1D atmospheric column at a time:

    compute_dep_coeffs(T, z, ne, vturb, vlos)
        GraphNet inference only -- fast (ms), no lightweaver dependency. This is the
        function to call inside the hot loop of an inversion.

    synthesis_lw(T, z, ne, vturb, vlos)
        Full lightweaver NLTE solve (iterates statistical equilibrium to convergence),
        exactly as used to build the training set. Slow -- use as ground truth / for
        validating the GraphNet, not inside an inversion loop.

    intensity_gnn(T, z, ne, vturb, vlos)
        NLTE-accelerated synthesis: GraphNet for the departure coefficients + a single
        lightweaver formal solution (no NLTE iteration). Same (wave, Iwave, log_dep)
        return signature as `synthesis_lw`, so it is a drop-in, much faster replacement.

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
    wave, Iwave, log_dep = api.intensity_gnn(T, z, ne, vturb, vlos)

Dependencies
------------
    compute_dep_coeffs   : numpy, torch, torch_geometric, torch_scatter
    synthesis_lw/intensity_gnn : the above, plus lightweaver (imported lazily, only
                                 when one of these two functions is actually called)
"""
import os
import sys

import numpy as np
import torch

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import graphnet

# Update this once the current training run finishes; any *_best.pth checkpoint trained
# with node_input_size=5 / edge_input_size=1 (T, z, ne, vturb, vlos node features, delta-z
# edge feature) works, just pass it explicitly via the `checkpoint=` argument.
DEFAULT_CHECKPOINT = os.path.join(_MODULE_DIR, 'checkpoints_si_v2/20260911-231808', '20260911-233633_best.pth')

# Normalization constants a checkpoint was trained with are read from the checkpoint file
# itself (Formal.py saves the Dataset.py NORM_STATS dict in use at training time as
# checkpoint['norm_stats']). This is the fallback for older checkpoints saved before that
# was added -- it reproduces the NORM_STATS / edge-feature formula Dataset.py used to use.
# Do not edit this to "fix" it: doing so would silently change predictions for every
# checkpoint trained before norm_stats was embedded.
_LEGACY_NORM_STATS = {
    'T_log10': {'mean': 4.115, 'std': 0.615},
    'z': {'mean': 1.331e6, 'std': 1.127e6},
    'tau_log10': {'mean': -7.822, 'std': 4.215},
    'ne_log10': {'mean': 17.529, 'std': 2.405},
    'vturb_km': {'mean': 2.068, 'std': 5.369},
    'vlos_km': {'mean': -0.689, 'std': 4.844},
    'delta_z': {'std': 1.127e6},  # old Dataset.py normalized edges by NORM_STATS['z']['std']
}

# Si I 10827 A window used to build the training set (generate_database.py).
DEFAULT_WAVE = np.linspace(1074.0, 1085.0, 1100)

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
        arr = np.asarray(val, dtype=np.float64)
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


def _load_model(checkpoint, device):
    """
    Load (and cache) a GraphNet checkpoint, along with the exact normalization constants it
    was trained with. Repeated calls are free after the first.
    """
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
    # Checkpoints saved before 'norm_stats' was added to Formal.py fall back to the
    # normalization Dataset.py used at the time -- the two MUST stay paired, since feeding a
    # checkpoint inputs normalized differently from its own training data silently degrades
    # predictions without any error.
    norm_stats = ckpt.get('norm_stats', _LEGACY_NORM_STATS)

    model = graphnet.EncodeProcessDecode(**hyperparams).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    _MODEL_CACHE[key] = (model, hyperparams, norm_stats)
    return _MODEL_CACHE[key]


def compute_dep_coeffs(T, z, ne, vturb, vlos, checkpoint=DEFAULT_CHECKPOINT, device='cpu'):
    """
    Predict NLTE departure coefficients for one atmospheric column with the GraphNet model.

    Pure GNN inference (no lightweaver call) -- milliseconds per column on CPU, safe to
    call many times per inversion iteration. The model is cached after the first call, so
    repeated calls with the same `checkpoint` only pay the load cost once per process.

    Parameters
    ----------
    T, z, ne, vturb, vlos : array_like, shape (n_depth,)
        Atmospheric stratification, see module docstring for units. `ne` is required: unlike
        a full lightweaver solve, the network cannot derive it on its own.
    checkpoint : str, optional
        Path to a trained GraphNet checkpoint (*_best.pth). Selects which training run to use.
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
        `checkpoint` does not exist.
    NotImplementedError
        `checkpoint` was trained with an unsupported node/edge feature configuration.
    """
    T, z, ne, vturb, vlos = _validate_atmosphere(T, z, ne, vturb, vlos)
    model, _, norm_stats = _load_model(checkpoint, device)

    node, edge_index, edge_attr = _build_graph(T, z, ne, vturb, vlos, norm_stats)
    u = torch.zeros((1, 1), dtype=torch.float32, device=device)
    batch = torch.zeros(node.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(node.to(device), edge_attr.to(device), edge_index.to(device), u, batch)

    log_dep = (out.cpu().numpy() * 5.0).T  # (n_depth, n_levels) -> (n_levels, n_depth)

    if not np.all(np.isfinite(log_dep)):
        raise RuntimeError("GraphNet produced non-finite departure coefficients")

    return log_dep


def _build_atmosphere(T, z, ne, vturb, vlos):
    """Shared lightweaver atmosphere/population setup for synthesis_lw and intensity_gnn."""
    import lightweaver as lw
    from lightweaver.rh_atoms import (H_6_atom, C_atom, OI_ord_atom, Si_atom_custom, Al_atom,
                                       CaII_atom, Fe_atom, He_9_atom, MgII_atom, N_atom, Na_atom, S_atom)

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
        Cosine of the viewing angle. Defaults to the outermost angle of the 5-point
        Gauss-Legendre quadrature (near disk-center), matching training-set generation.
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
    supplied) + a single lightweaver formal solution (no NLTE iteration). Returns the same
    (wave, Iwave, log_dep) triple as `synthesis_lw`, so the two are interchangeable -- this
    is the fast function to use inside an inversion.

    Parameters
    ----------
    T, z, ne, vturb, vlos : array_like, shape (n_depth,)
        See module docstring for units.
    log_dep : ndarray, shape (n_levels, n_depth), optional
        Precomputed departure coefficients (e.g. from a prior `compute_dep_coeffs` call),
        to avoid recomputing them. If None (default), they are predicted here.
    wave, mu : optional
        See `synthesis_lw`.
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
    FileNotFoundError, NotImplementedError, RuntimeError
        Only if `log_dep` is not supplied: propagated from the internal
        `compute_dep_coeffs` call, see there.
    """
    T, z, ne, vturb, vlos = _validate_atmosphere(T, z, ne, vturb, vlos)
    if wave is None:
        wave = DEFAULT_WAVE
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
    Iwave = ctx.compute_rays(wave, [mu], stokes=False)

    return np.asarray(wave), np.asarray(Iwave), log_dep


if __name__ == '__main__':
    # Lightweight smoke test for compute_dep_coeffs (no lightweaver needed). Uses a
    # simple isothermal-ish toy column purely to exercise the code path end-to-end.
    n = 60
    z = np.linspace(-5e5, 2.5e6, n)                    # m
    T = 6000.0 + 4000.0 * np.exp(-((z - 1e6) / 8e5)**2)  # K
    ne = 10.0 ** np.linspace(21.0, 17.0, n)             # m^-3
    vturb = np.full(n, 2.0e3)                            # m/s
    vlos = np.zeros(n)                                   # m/s

    log_dep = compute_dep_coeffs(T, z, ne, vturb, vlos)
    print('log_dep shape:', log_dep.shape)
    print('log_dep range:', log_dep.min(), log_dep.max())
    print('compute_dep_coeffs smoke test passed')
