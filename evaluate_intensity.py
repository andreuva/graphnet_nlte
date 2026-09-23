"""
Spectral acceptance test for the Si I GraphNet, measured in the emergent Si I 1083.0 nm profile.

Why this script exists
----------------------
The training loss (masked MSE on log10(b)/5) is a poor proxy for what the network is for. It
weights every level and depth point alike, while the emergent profile is set by two levels
(the 3s2 3p 4s 3PO J=2 lower and 3s2 3p 4p 3PE J=2 upper level of the transition) over the few
hundred km where the line-core contribution function is non-zero. A checkpoint can improve its
MSE without improving a single synthesised profile, and vice versa.

What is measured, and against what
----------------------------------
For every column four profiles are synthesised on a common fine grid (2 pm, +-0.6 nm):

    converged : full lightweaver NLTE solve (statistical equilibrium iterated to convergence,
                J self-consistent). This is the ground truth an inversion would compute.
    formal    : the stored, converged departure coefficients + the api.intensity_gnn synthesis
                (one formal solution for the mean radiation field, then the emergent rays; no
                statistical-equilibrium iteration). Perfect populations, deployed solver.
    graphnet  : the network's departure coefficients + the same synthesis (the deployed path).
    lte       : b = 1 everywhere, same synthesis (no NLTE correction at all).

which gives a complete error budget:

    deployed error   graphnet - converged   what a user of api.intensity_gnn actually gets
    network error    graphnet - formal      the part due to the GraphNet alone
    solver floor     formal   - converged   the part due to skipping the NLTE iteration (the
                                            mean radiation field from one formal solution)
    LTE baseline     lte      - converged   the error of doing nothing; the skill of the network
                                            is measured against this

Metric catalogue (each is a per-column number; the report gives median with a bootstrap 95% CI,
the 90th and 99th percentiles and the maximum, so the tail is never hidden by the median):

  Spectral fidelity -- what an inversion code sees
    rms_norm    RMS of (I_test - I_ref)/I_c over the window, in units of the reference continuum
                (the mean intensity at the window edges, 0.57 nm from the core).
                Inversion codes (SIR, NICOLE, STiC, DeSIRe, ...) fit continuum-normalised Stokes
                profiles with a chi^2 whose weights are 1/sigma^2 in these units, so this is the
                quantity that must sit below the observational noise for the acceleration to be
                invisible to the inversion.
    rms_shape   Same, after normalising each profile to its OWN continuum. This removes any pure
                continuum offset and isolates the profile shape.
    max_norm    Worst-wavelength |dI|/I_c. core_norm: the same at the reference line core.
    chi2nu      Reduced chi^2 against the reference for a photon-noise level sigma (in I_c units;
                default 1e-3 for ground-based spectropolarimetry, and 3e-4 for deep DKIST-class
                integrations). chi2nu <= 1 means the profile is indistinguishable from the truth
                at that S/N. The fraction of columns passing each level is reported.
    rel_median, rel_max  |dI/I| statistics, the metric used by Chappell & Pereira (2022) and
                Vicente Arevalo et al. (2022) for neural NLTE departure-coefficient emulators.
    cont_rel    Relative error of the continuum itself.

  Line parameters -- what an inversion derives
    depth_err   Error of the line depth 1 - I_core/I_c (I_c units).
    ew_rel      Relative equivalent-width error.
    v_bias_ms   Doppler-velocity bias of the line core [m/s], from a parabolic sub-pixel fit of
                the core. A direct bias on the line-of-sight velocity an inversion would infer.
    fwhm_err_pm Error of the full width at half depth [pm], a proxy for the thermal /
                microturbulent / magnetic broadening an inversion would infer.

  Skill against LTE (Murphy 1988 skill score):  S = 1 - median(err_graphnet)/median(err_lte),
    plus the paired win fraction (columns where the network beats LTE) and the distribution of
    the per-column ratio err_graphnet/err_lte.

  Departure coefficients where the line forms
    The line-core intensity contribution function of the reference, C_I(z) = chi S exp(-tau/mu)/mu
    (Hubeny & Mihalas 2014, ch. 11), weights the errors of log10 b_lower (line opacity),
    log10 b_upper and log10(b_upper/b_lower) (line source function, S_L/B ~ b_u/b_l). The
    unweighted masked MAE over all 16 levels is given too, for comparison with the training
    metric, together with the population-conservation error |sum_i n_i/N - 1| at the formation
    height and the shift of the tau_core = 1 height the network induces.

  Stratification and failures
    All headline metrics are also given per column origin (Bifrost snapshot columns, 211 depth
    points, versus perturbed semi-empirical reference models) and the worst columns are listed
    by index so they can be inspected.

  Acceleration
    Wall-clock time of the NLTE iteration the network replaces, of the network itself (batched,
    on --device) and of the formal solution and atmosphere setup that both paths share.

  Consistency of the pipeline
    The converged solve is compared to the stored departure coefficients and to the stored
    emergent intensity of the database, to prove the reference here IS the database's truth.

Inputs
------
The network output can come from two places:
    --ck   a checkpoint (*.pth or run directory): the model is run here, through api._build_graph,
           i.e. the exact deployed feature construction;
    --pred a prediction pickle written by test_prediction.py / Formal.test (validation_checkpoint_
           *.pkl): its `prediction` list is used as-is, so this evaluates exactly the numbers
           explore_tests.py plots. Predictions are matched to the split by position (Formal.test
           runs the loader with shuffle=False).

Use the `validation` split (Bifrost snap530, a different snapshot from train/test).

Usage
-----
    python evaluate_intensity.py --rd ../data_1d_si_v3/ --ck checkpoints_si_v3/20260923-171902 --n 1000
    python evaluate_intensity.py --rd ../data_1d_si_v3/ \
        --pred checkpoints_si_v3/20260923-171902/validation_checkpoint_20260923-171902_at_20260923-215654.pkl

Outputs (in --sav, default <run dir>/acceptance/): a text report, a JSON with the headline
numbers (for tracking checkpoints), a pickle with every per-column row and residual spectrum,
and the figures acceptance_*_{profiles,error_spectrum,distributions,line_parameters,
departure,stratified}.png.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

# ----------------------------------------------------------------------------- constants
# Vacuum wavelength of the Si I 3s2 3p 4p 3PE J=2 -> 3s2 3p 4s 3PO J=2 transition,
# 1e7 / (49188.617 - 39955.053) cm-1. Usually quoted by its air wavelength, 10827 A.
LINE_CORE_NM = 1083.0038
LOWER_LEVEL, UPPER_LEVEL = 8, 11            # 0-based indices in si_atom.Si_atom_custom
C_LIGHT_MS = 299792458.0

# Synthesis grid: 2 pm sampling over +-0.6 nm. The Si Doppler width is 4.4 pm at the coolest
# temperature in the training set (2500 K, vturb = 0), so the core is always resolved.
WAVE_LO, WAVE_HI, WAVE_N = 1082.4, 1083.6, 601

# The continuum is the mean intensity over this many points at each edge of the window (30 pm per
# side, 0.57 nm from the core): robust to Doppler shifts of up to ~100 km/s and, unlike the window
# maximum, correct for profiles with an emission core.
N_EDGE = 15

# Photon-noise levels, in units of the continuum, at which the profiles are judged.
DEFAULT_NOISE = (1e-3, 3e-4)

# Loss mask threshold of Dataset.py: levels/depths with n_i/N below this are unsupervised.
NEGLIGIBLE_LOG_N_OVER_NTOT = -9.0

# Column origin: Bifrost columns carry 211 depth points, the perturbed semi-empirical reference
# atmospheres 80-102 (see generate_database.py / README).
BIFROST_NDEPTH = 211

TRIALS = ('graphnet', 'network', 'floor', 'lte')
TRIAL_LABEL = {'graphnet': 'GraphNet (deployed)', 'network': 'GraphNet vs formal (network only)',
               'floor': 'formal solution floor (no SE iteration)', 'lte': 'LTE (no NLTE correction)'}

# Colours: neutral ink for the reference (truth, not a series) and the first three slots of a
# colour-vision-deficiency validated categorical palette for the three competing series.
COLOR = {'reference': '#2B3A3C', 'graphnet': '#2a78d6', 'lte': '#eb6834', 'floor': '#1baf7a',
         'network': '#2a78d6', 'grid': '#8a9a9c', 'muted': '#5C6E70', 'bifrost': '#4a3aa7',
         'semi-empirical': '#eda100'}

_trapz = getattr(np, 'trapezoid', None) or np.trapz


# ----------------------------------------------------------------------------- data
def load_split(datadir, prefix):
    """Read one split. `wave`/`Iwave` (the stored converged spectra) are optional."""
    out = {}
    for name in ('T', 'z', 'ne', 'vturb', 'vlos', 'logdeparture', 'n_Nat'):
        path = os.path.join(datadir, f'{prefix}_{name}.pkl')
        if name == 'n_Nat' and not os.path.isfile(path):
            out[name] = None
            continue
        with open(path, 'rb') as fh:
            out[name] = pickle.load(fh)
    for name in ('wave', 'Iwave'):
        path = os.path.join(datadir, f'{prefix}_{name}.pkl')
        out[name] = None
        if os.path.isfile(path):
            with open(path, 'rb') as fh:
                out[name] = pickle.load(fh)
    if out['wave'] is not None:
        out['wave'] = np.asarray(out['wave'], dtype=np.float64)
    return out


def column_arrays(data, idx):
    """The five atmospheric arrays of one column as contiguous float64 (lightweaver needs that)."""
    return tuple(np.ascontiguousarray(data[k][idx], dtype=np.float64)
                 for k in ('T', 'z', 'ne', 'vturb', 'vlos'))


def load_predictions(path):
    """Prediction pickle from Formal.test: scaled (n_depth, 16) arrays -> clipped (16, n_depth)."""
    with open(path, 'rb') as fh:
        P = pickle.load(fh)
    preds = [np.clip(np.asarray(p, dtype=np.float64).T * 5.0, -10.0, 10.0) for p in P['prediction']]
    return preds, P


def predict_with_model(checkpoint, data, indices, device, batch_size):
    """
    Run the checkpoint over the selected columns through the deployed feature construction
    (api._build_graph), in batches, and time it. Returns (predictions dict idx->(16,n), timing).
    """
    import torch
    import api

    model, hyperparams, norm_stats = api._load_model(checkpoint, device)
    is_cuda = str(device).startswith('cuda')

    def forward(cols):
        nodes, eidx, eattr, bvec, lengths, offset = [], [], [], [], [], 0
        for g, (T, z, ne, vturb, vlos) in enumerate(cols):
            node, edge_index, edge_attr = api._build_graph(T, z, ne, vturb, vlos, norm_stats)
            nodes.append(node)
            eidx.append(edge_index + offset)
            eattr.append(edge_attr)
            bvec.append(torch.full((len(T),), g, dtype=torch.long))
            lengths.append(len(T))
            offset += len(T)
        node = torch.cat(nodes).to(device)
        edge_index = torch.cat(eidx, dim=1).to(device)
        edge_attr = torch.cat(eattr).to(device)
        batch = torch.cat(bvec).to(device)
        u = torch.zeros((len(cols), 1), dtype=torch.float32, device=device)
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(node, edge_attr, edge_index, u, batch)
        if is_cuda:
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        out = out.cpu().numpy().astype(np.float64) * 5.0
        preds, start = [], 0
        for n in lengths:
            preds.append(np.clip(out[start:start + n].T, -10.0, 10.0))
            start += n
        return preds, dt

    # Warm-up (kernel/autotune costs must not land in the timing).
    forward([column_arrays(data, int(indices[0]))])

    predictions, per_col_batched = {}, []
    for s in range(0, len(indices), batch_size):
        chunk = [int(i) for i in indices[s:s + batch_size]]
        preds, dt = forward([column_arrays(data, i) for i in chunk])
        per_col_batched.append(dt / len(chunk))
        for i, p in zip(chunk, preds):
            predictions[i] = p

    single = []
    for i in indices[:min(8, len(indices))]:
        _, dt = forward([column_arrays(data, int(i))])
        single.append(dt)

    timing = {'gnn_per_column_batched_s': float(np.median(per_col_batched)),
              'gnn_single_column_s': float(np.median(single)),
              'gnn_batch_size': int(batch_size), 'gnn_device': str(device),
              'n_parameters': int(sum(p.numel() for p in model.parameters()))}
    return predictions, dict(hyperparams), timing


# ----------------------------------------------------------------------------- synthesis (worker)
_W = {}


def _worker_init(wave):
    import lightweaver as lw
    import api
    _W['lw'], _W['api'], _W['wave'] = lw, api, np.asarray(wave, dtype=np.float64)


def _formal_solution(atmos, spect, eqPops, log_dep, nstar, mu, wave, depth_data):
    """
    The api.intensity_gnn synthesis with the Si populations set from log_dep (LTE if None): one
    formal solution on the full grid to fill the mean radiation field (a fresh Context has
    J = 0), then the emergent rays; no statistical-equilibrium iteration. Optionally returns
    the emergent-ray opacity and emissivity at every depth (for the contribution function and
    the tau = 1 height).
    """
    lw = _W['lw']
    pops = nstar if log_dep is None else (10.0 ** log_dep) * nstar
    eqPops.atomicPops['Si'].n[:] = pops
    t0 = time.perf_counter()
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
    ctx.formal_sol_gamma_matrices()
    I, rc = ctx.compute_rays(wave, [mu], stokes=False, returnCtx=True)
    dt = time.perf_counter() - t0
    chi = eta = None
    if depth_data:
        rc.depthData.fill = True
        rc.formal_sol_gamma_matrices()
        # depthData arrays are (Nwave, Nmu, 2, Nspace); index 1 is the outgoing (emergent) ray,
        # whose I at the top equals the emergent intensity. Verified against compute_rays.
        chi = np.array(rc.depthData.chi)[:, 0, 1, :]
        eta = np.array(rc.depthData.eta)[:, 0, 1, :]
    return np.asarray(I, dtype=np.float64), chi, eta, dt


def synthesise_column(task):
    """
    All the radiative transfer for one column. Returns raw arrays only; the metrics are
    computed in the parent so they can be changed without re-running the synthesis.
    """
    idx = task['index']
    try:
        lw, api, wave = _W['lw'], _W['api'], _W['wave']
        T, z, ne, vturb, vlos = task['atmos']
        vturb_phys = np.maximum(vturb, 0.0)     # only vturb**2 reaches the physics
        dep_ref, dep_gnn = task['dep_ref'], task['dep_gnn']

        t0 = time.perf_counter()
        atmos, aSet, spect, eqPops = api._build_atmosphere(T, z, ne, vturb_phys, vlos)
        eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
        t_build = time.perf_counter() - t0
        mu = float(atmos.muz[-1]) if task['mu'] is None else float(task['mu'])
        si = eqPops.atomicPops['Si']
        nstar = np.array(si.nStar, dtype=np.float64)
        ntot = np.array(si.nTotal, dtype=np.float64)

        # --- converged reference: exactly api.synthesis_lw
        t0 = time.perf_counter()
        ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
        lw.iterate_ctx_se(ctx, prd=False, quiet=True)
        eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
        ctx.formal_sol_gamma_matrices()
        t_iterate = time.perf_counter() - t0
        dep_conv = np.log10(np.asarray(ctx.activeAtoms[0].n) / np.asarray(ctx.activeAtoms[0].nStar))
        converged = bool(np.all(np.isfinite(dep_conv)))
        I_conv = np.asarray(ctx.compute_rays(wave, [mu], stokes=False), dtype=np.float64)
        I_conv_stored = None
        if task['stored_wave'] is not None:
            I_conv_stored = np.asarray(ctx.compute_rays(task['stored_wave'], [mu], stokes=False),
                                       dtype=np.float64)
        del ctx

        # --- deployed-path syntheses, reusing the atmosphere; nStar is untouched by Context
        #     construction, so writing n[:] in place is equivalent to a fresh build.
        I_form, chi_ref, eta_ref, t_f1 = _formal_solution(atmos, spect, eqPops, dep_ref, nstar, mu, wave, True)
        I_gnn, chi_gnn, _, t_f2 = _formal_solution(atmos, spect, eqPops, dep_gnn, nstar, mu, wave, True)
        I_lte, _, _, t_f3 = _formal_solution(atmos, spect, eqPops, None, nstar, mu, wave, False)

        core = int(np.argmin(I_form))
        cont = int(np.argmax(I_form))
        return {
            'index': idx, 'ok': True, 'converged': converged, 'mu': mu,
            'z': np.asarray(atmos.z, dtype=np.float64),
            'I_conv': I_conv, 'I_form': I_form, 'I_gnn': I_gnn, 'I_lte': I_lte,
            'I_conv_stored': I_conv_stored,
            'chi_core_ref': chi_ref[core], 'eta_core_ref': eta_ref[core],
            'chi_cont_ref': chi_ref[cont], 'chi_core_gnn': chi_gnn[core],
            'nstar': nstar, 'ntot': ntot,
            'dep_conv_vs_stored_max': float(np.nanmax(np.abs(dep_conv - dep_ref))) if converged else np.nan,
            't_build': t_build, 't_iterate': t_iterate, 't_formal': float(np.median([t_f1, t_f2, t_f3])),
        }
    except Exception as exc:                                       # noqa: BLE001
        return {'index': idx, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


# ----------------------------------------------------------------------------- metrics
def parabolic_minimum(wave, I, lo=0, hi=None):
    """Sub-pixel position and value of the profile minimum (3-point parabola), searched in [lo, hi)."""
    hi = len(I) if hi is None else hi
    k = lo + int(np.argmin(I[lo:hi]))
    k = int(np.clip(k, 1, len(I) - 2))
    y0, y1, y2 = I[k - 1], I[k], I[k + 1]
    denom = y0 - 2.0 * y1 + y2
    h = wave[1] - wave[0]
    if denom <= 0:
        return float(wave[k]), float(y1)
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(wave[k] + delta * h), float(y1 - 0.25 * (y0 - y2) * delta)


def half_width(wave, I, Ic, core_idx, half_level):
    """Full width at half depth by linear interpolation of the two crossings around the core."""
    left = right = np.nan
    for k in range(core_idx, 0, -1):
        if I[k - 1] >= half_level > I[k] or I[k - 1] >= half_level >= I[k]:
            left = wave[k] + (half_level - I[k]) * (wave[k - 1] - wave[k]) / (I[k - 1] - I[k])
            break
    for k in range(core_idx, len(I) - 1):
        if I[k + 1] >= half_level > I[k] or I[k + 1] >= half_level >= I[k]:
            right = wave[k] + (half_level - I[k]) * (wave[k + 1] - wave[k]) / (I[k + 1] - I[k])
            break
    return float(right - left)


# The core of a trial profile is searched within this distance of the reference core (about
# 40 km/s), so a distorted or very shallow trial profile cannot hand its minimum to the wing.
CORE_SEARCH_NM = 0.15

# Below this reference line depth there is no core to locate: the core velocity and FWHM of such
# columns are undefined and are reported as NaN (they still count in every intensity metric).
MIN_DEPTH_FOR_CORE = 0.02


def continuum_level(I):
    """Continuum intensity: mean of the outermost N_EDGE points on each side of the window."""
    return float(0.5 * (np.mean(I[:N_EDGE]) + np.mean(I[-N_EDGE:])))


def line_parameters(wave, I, Ic, center_nm=None):
    """
    Continuum, depth, equivalent width, core position/velocity and FWHM of one profile. With
    `center_nm` the core is searched only within +-CORE_SEARCH_NM of that wavelength.
    """
    lo, hi = 0, len(I)
    if center_nm is not None:
        lo = int(np.searchsorted(wave, center_nm - CORE_SEARCH_NM))
        hi = max(lo + 3, int(np.searchsorted(wave, center_nm + CORE_SEARCH_NM)))
    lam_core, I_core = parabolic_minimum(wave, I, lo, hi)
    depth = 1.0 - I_core / Ic
    ew_pm = float(_trapz(1.0 - I / Ic, wave) * 1e3)
    core_idx = lo + int(np.argmin(I[lo:hi]))
    fwhm_pm = half_width(wave, I, Ic, core_idx, Ic - 0.5 * (Ic - I_core)) * 1e3
    v_ms = C_LIGHT_MS * (lam_core - LINE_CORE_NM) / LINE_CORE_NM
    return {'continuum': float(Ic), 'depth': float(depth), 'ew_pm': ew_pm, 'core_nm': lam_core,
            'v_ms': float(v_ms), 'fwhm_pm': float(fwhm_pm), 'peak': float(np.max(I) / Ic - 1.0)}


def profile_errors(wave, I_test, I_ref, noise_levels):
    """Spectral and line-parameter errors of one trial profile against a reference profile."""
    Ic_ref, Ic_test = continuum_level(I_ref), continuum_level(I_test)
    r = (I_test - I_ref) / Ic_ref                       # I_c units, common continuum
    r_shape = I_test / Ic_test - I_ref / Ic_ref         # each on its own continuum
    rel = np.abs(I_test - I_ref) / np.abs(I_ref)
    core_idx = int(np.argmin(I_ref))
    p_ref = line_parameters(wave, I_ref, Ic_ref)
    p_test = line_parameters(wave, I_test, Ic_test, center_nm=p_ref['core_nm'])
    out = {
        'rms_norm': float(np.sqrt(np.mean(r ** 2))),
        'rms_shape': float(np.sqrt(np.mean(r_shape ** 2))),
        'max_norm': float(np.max(np.abs(r))),
        'core_norm': float(abs(r[core_idx])),
        'cont_rel': float(Ic_test / Ic_ref - 1.0),
        'rel_median': float(np.median(rel)),
        'rel_max': float(np.max(rel)),
        'depth_err': float(p_test['depth'] - p_ref['depth']),
        'ew_rel': float((p_test['ew_pm'] - p_ref['ew_pm']) / p_ref['ew_pm']) if p_ref['ew_pm'] != 0 else np.nan,
        'v_bias_ms': float(p_test['v_ms'] - p_ref['v_ms']),
        'fwhm_err_pm': float(p_test['fwhm_pm'] - p_ref['fwhm_pm']),
        'fwhm_rel': float((p_test['fwhm_pm'] - p_ref['fwhm_pm']) / p_ref['fwhm_pm']) if p_ref['fwhm_pm'] > 0 else np.nan,
        'peak_err': float(p_test['peak'] - p_ref['peak']),
    }
    if p_ref['depth'] < MIN_DEPTH_FOR_CORE:
        out['v_bias_ms'] = out['fwhm_err_pm'] = out['fwhm_rel'] = np.nan
    for s in noise_levels:
        out[f'chi2nu_{s:g}'] = float(np.mean(r ** 2) / s ** 2)
    return out, r.astype(np.float32)


def optical_depth(chi, z, mu):
    """Monochromatic optical depth along the emergent ray, from the top (z decreasing with index)."""
    dz = -np.diff(z)
    return np.concatenate([[0.0], np.cumsum(0.5 * (chi[1:] + chi[:-1]) * dz)]) / mu


def formation_height(tau, z):
    """Height where tau = 1 (linear interpolation; NaN if the column is optically thin)."""
    if tau[-1] < 1.0:
        return np.nan
    return float(np.interp(1.0, tau, z))


def contribution_weights(chi, eta, z, mu):
    """
    Emergent-intensity contribution function C_I(z) = chi S exp(-tau/mu)/mu of one wavelength,
    returned as normalised depth weights (trapezoid rule) plus the raw C_I.
    """
    tau = optical_depth(chi, z, mu)
    CF = eta * np.exp(-tau) / mu                 # chi * (eta/chi) * exp(-tau) / mu
    dz = np.abs(np.gradient(z))
    w = np.clip(CF * dz, 0.0, None)
    total = w.sum()
    w = w / total if total > 0 else np.full_like(w, 1.0 / len(w))
    return w, CF, tau


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(np.interp(q, cw / cw[-1], x[order]))


def column_metrics(res, task, wave, noise_levels):
    """Assemble the per-column row from the worker's raw arrays."""
    I_conv, I_form, I_gnn, I_lte = res['I_conv'], res['I_form'], res['I_gnn'], res['I_lte']
    z, mu = res['z'], res['mu']
    dep_ref, dep_gnn = task['dep_ref'], task['dep_gnn']
    dep_clip = np.clip(dep_ref, -10.0, 10.0)
    n = len(z)

    row = {'index': res['index'], 'n_depth': n, 'converged': res['converged'], 'mu': mu,
           'origin': 'bifrost' if n == BIFROST_NDEPTH else 'semi-empirical',
           'reference': line_parameters(wave, I_conv, continuum_level(I_conv))}
    residuals = {}
    pairs = {'graphnet': (I_gnn, I_conv), 'network': (I_gnn, I_form),
             'floor': (I_form, I_conv), 'lte': (I_lte, I_conv)}
    for name, (I_test, I_ref) in pairs.items():
        row[name], residuals[name] = profile_errors(wave, I_test, I_ref, noise_levels)

    # --- where the line forms (reference populations, single formal solution)
    w, CF, tau_core = contribution_weights(res['chi_core_ref'], res['eta_core_ref'], z, mu)
    tau_gnn = optical_depth(res['chi_core_gnn'], z, mu)
    tau_cont = optical_depth(res['chi_cont_ref'], z, mu)
    z_tau1_ref = formation_height(tau_core, z)
    z_tau1_gnn = formation_height(tau_gnn, z)
    z_tau1_cont = formation_height(tau_cont, z)
    row['formation'] = {
        'z_tau1_core_km': z_tau1_ref / 1e3, 'z_tau1_core_gnn_km': z_tau1_gnn / 1e3,
        'z_tau1_cont_km': z_tau1_cont / 1e3, 'z_tau1_shift_km': (z_tau1_gnn - z_tau1_ref) / 1e3,
        'z_cf_5_km': weighted_quantile(z, w, 0.05) / 1e3, 'z_cf_95_km': weighted_quantile(z, w, 0.95) / 1e3,
        'z_cf_peak_km': float(z[np.argmax(CF)]) / 1e3,
        'T_cf_km': float(np.sum(w * task['atmos'][0])),
        'vlos_cf_kms': float(np.sum(w * task['atmos'][4])) / 1e3,
    }

    # --- departure coefficients, weighted by the line-core contribution function
    def dep_errors(dep):
        d = dep - dep_clip
        ratio = (dep[UPPER_LEVEL] - dep[LOWER_LEVEL]) - (dep_clip[UPPER_LEVEL] - dep_clip[LOWER_LEVEL])
        out = {'cf_mae_lower': float(np.sum(w * np.abs(d[LOWER_LEVEL]))),
               'cf_mae_upper': float(np.sum(w * np.abs(d[UPPER_LEVEL]))),
               'cf_mae_ratio': float(np.sum(w * np.abs(ratio))),
               'cf_bias_lower': float(np.sum(w * d[LOWER_LEVEL])),
               'cf_bias_ratio': float(np.sum(w * ratio)),
               'mae_all': float(np.mean(np.abs(d)))}
        if task['mask'] is not None and task['mask'].any():
            out['mae_masked'] = float(np.mean(np.abs(d[task['mask']])))
        else:
            out['mae_masked'] = out['mae_all']
        # population conservation, sum_i n_i / N_Si - 1, at the formation height
        cons = np.abs((10.0 ** dep * res['nstar']).sum(axis=0) / res['ntot'] - 1.0)
        out['pop_cons_cf'] = float(np.sum(w * cons))
        return out

    row['dep_graphnet'] = dep_errors(dep_gnn)
    row['dep_lte'] = dep_errors(np.zeros_like(dep_gnn))
    # b_lower and b_upper/b_lower themselves at the formation height, so that the size of the
    # NLTE effect the network has to reproduce is on record next to its error.
    row['dep_reference'] = {'cf_log_b_lower': float(np.sum(w * dep_clip[LOWER_LEVEL])),
                            'cf_log_b_ratio': float(np.sum(w * (dep_clip[UPPER_LEVEL] - dep_clip[LOWER_LEVEL])))}

    # --- pipeline consistency
    row['consistency'] = {'dep_conv_vs_stored_max': res['dep_conv_vs_stored_max']}
    if res['I_conv_stored'] is not None and task['stored_Iwave'] is not None:
        rel = np.abs(res['I_conv_stored'] / task['stored_Iwave'] - 1.0)
        row['consistency']['I_conv_vs_stored_median'] = float(np.median(rel))
        row['consistency']['I_conv_vs_stored_max'] = float(np.max(rel))

    row['timing'] = {'t_build': res['t_build'], 't_iterate': res['t_iterate'], 't_formal': res['t_formal']}
    profile = {'I_conv': I_conv.astype(np.float32), 'I_form': I_form.astype(np.float32),
               'I_gnn': I_gnn.astype(np.float32), 'I_lte': I_lte.astype(np.float32),
               'z': z.astype(np.float32), 'cf_core': CF.astype(np.float32), 'w_cf': w.astype(np.float32),
               'dep_ref_lower': dep_clip[LOWER_LEVEL].astype(np.float32),
               'dep_gnn_lower': dep_gnn[LOWER_LEVEL].astype(np.float32),
               'dep_ref_ratio': (dep_clip[UPPER_LEVEL] - dep_clip[LOWER_LEVEL]).astype(np.float32),
               'dep_gnn_ratio': (dep_gnn[UPPER_LEVEL] - dep_gnn[LOWER_LEVEL]).astype(np.float32)}
    return row, residuals, profile


# ----------------------------------------------------------------------------- aggregation
def values(rows, key, field, absolute=False):
    v = np.array([r[key][field] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return np.abs(v) if absolute else v


def median_ci(x, rng, n_boot=2000, level=0.95):
    """Median and bootstrap percentile confidence interval."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    if x.size == 1:
        return float(x[0]), (float(x[0]), float(x[0]))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    a = (1.0 - level) / 2.0
    return float(np.median(x)), (float(np.quantile(meds, a)), float(np.quantile(meds, 1.0 - a)))


def tail(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {'p50': np.nan, 'p90': np.nan, 'p99': np.nan, 'max': np.nan}
    return {'p50': float(np.median(x)), 'p90': float(np.quantile(x, 0.90)),
            'p99': float(np.quantile(x, 0.99)), 'max': float(np.max(x))}


def skill(rows, field, absolute=True):
    """Skill score, win fraction and the per-column error ratio of GraphNet against LTE."""
    g = np.array([r['graphnet'][field] for r in rows], dtype=float)
    l = np.array([r['lte'][field] for r in rows], dtype=float)
    if absolute:
        g, l = np.abs(g), np.abs(l)
    ok = np.isfinite(g) & np.isfinite(l)
    g, l = g[ok], l[ok]
    if g.size == 0:
        return {'skill': np.nan, 'win_fraction': np.nan, 'ratio_p50': np.nan, 'ratio_p90': np.nan}
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = g / l
    ratio = ratio[np.isfinite(ratio)]
    med_l = np.median(l)
    return {'skill': float(1.0 - np.median(g) / med_l) if med_l > 0 else np.nan,
            'win_fraction': float(np.mean(g < l)),
            'ratio_p50': float(np.median(ratio)) if ratio.size else np.nan,
            'ratio_p90': float(np.quantile(ratio, 0.90)) if ratio.size else np.nan}


def fmt(x, spec='{:.3g}'):
    return '   nan' if x is None or not np.isfinite(x) else spec.format(x)


def build_report(rows, args, checkpoint, source, timing, hyperparams, noise_levels, rng):
    L = []
    P = L.append
    n = len(rows)
    n_bif = sum(r['origin'] == 'bifrost' for r in rows)
    P('=' * 96)
    P(f'Si I {LINE_CORE_NM:.4f} nm spectral acceptance test -- {n} columns of "{args.dtst}" ({args.rd})')
    P(f'network output : {source}')
    P(f'checkpoint     : {checkpoint}')
    if hyperparams:
        P(f'architecture   : latent {hyperparams.get("latent_size")}, {hyperparams.get("n_message_passing_steps")} '
          f'message-passing steps, MLP {hyperparams.get("mlp_n_hidden_layers")}x{hyperparams.get("mlp_hidden_size")}')
    P(f'columns        : {n_bif} Bifrost (snap530, {BIFROST_NDEPTH} pts), {n - n_bif} perturbed semi-empirical models; '
      f'mu = {np.median([r["mu"] for r in rows]):.4f}; window {WAVE_LO}-{WAVE_HI} nm, {WAVE_N} pts')
    d = values(rows, 'reference', 'depth')
    P(f'reference line : depth {np.median(d):.3f} (p10 {np.quantile(d, .1):.3f}, p90 {np.quantile(d, .9):.3f}), '
      f'EW {np.median(values(rows, "reference", "ew_pm")):.1f} pm, FWHM {np.median(values(rows, "reference", "fwhm_pm")):.1f} pm; '
      f'NLTE at formation: log10 b_lower {np.median(values(rows, "dep_reference", "cf_log_b_lower")):+.3f}, '
      f'log10 b_u/b_l {np.median(values(rows, "dep_reference", "cf_log_b_ratio")):+.3f} (medians)')
    P('=' * 96)

    # ---- 1. error budget
    P('\n1. ERROR BUDGET  --  RMS of (I - I_ref)/I_c over the window, in units of the continuum')
    P(f'   {"profile":<36s} {"median [95% CI]":>26s} {"p90":>9s} {"p99":>9s} {"max":>9s}')
    for name in TRIALS:
        x = values(rows, name, 'rms_norm')
        m, (lo, hi) = median_ci(x, rng)
        t = tail(x)
        P(f'   {TRIAL_LABEL[name]:<36s} {fmt(m):>9s} [{fmt(lo):>7s}, {fmt(hi):>7s}] {fmt(t["p90"]):>9s} {fmt(t["p99"]):>9s} {fmt(t["max"]):>9s}')
    P('   same, each profile on its own continuum (shape only):')
    for name in TRIALS:
        t = tail(values(rows, name, 'rms_shape'))
        P(f'   {TRIAL_LABEL[name]:<36s} {fmt(t["p50"]):>26s} {fmt(t["p90"]):>9s} {fmt(t["p99"]):>9s} {fmt(t["max"]):>9s}')
    P('   worst wavelength |dI|/I_c and line-core |dI|/I_c (medians):')
    for name in TRIALS:
        P(f'   {TRIAL_LABEL[name]:<36s} max {fmt(np.median(values(rows, name, "max_norm"))):>9s}   '
          f'core {fmt(np.median(values(rows, name, "core_norm"))):>9s}   '
          f'|dI/I| median {fmt(np.median(values(rows, name, "rel_median"))):>9s}   '
          f'continuum {fmt(np.median(values(rows, name, "cont_rel", absolute=True))):>9s}')

    # ---- 2. noise test
    P('\n2. OBSERVATIONAL-NOISE TEST  --  is the error hidden below the photon noise?')
    P(f'   {"noise sigma/I_c":<18s} {"profile":<36s} {"RMS < sigma":>12s} {"chi2_nu <= 1":>13s} {"median chi2_nu":>15s}')
    for s in noise_levels:
        for name in TRIALS:
            rms = values(rows, name, 'rms_norm')
            c2 = values(rows, name, f'chi2nu_{s:g}')
            P(f'   {s:<18.1e} {TRIAL_LABEL[name]:<36s} {100 * np.mean(rms < s):>11.1f}% {100 * np.mean(c2 <= 1):>12.1f}% {fmt(np.median(c2)):>15s}')
    g = values(rows, 'graphnet', 'rms_norm')
    P(f'   noise-equivalent error of the deployed GraphNet: sigma = {np.median(g):.2e} I_c (median), '
      f'{np.quantile(g, 0.99):.2e} I_c (99th percentile)')

    # ---- 3. line parameters
    P('\n3. LINE PARAMETERS  --  what an inversion would infer  (median |error|; skill = 1 - GNN/LTE)')
    n_core = int(np.sum(values(rows, 'reference', 'depth') >= MIN_DEPTH_FOR_CORE))
    P(f'   (core velocity and FWHM over the {n_core} columns with a reference line depth >= {MIN_DEPTH_FOR_CORE})')
    P(f'   {"parameter":<30s} {"GraphNet":>11s} {"LTE":>11s} {"skill":>7s} {"GNN wins":>9s} {"GNN/LTE p50":>12s} {"p90":>8s}')
    for field, label, spec in (('depth_err', 'line depth [I_c]', '{:.4f}'), ('ew_rel', 'equivalent width [rel]', '{:.4f}'),
                               ('v_bias_ms', 'core velocity [m/s]', '{:.1f}'), ('fwhm_err_pm', 'FWHM [pm]', '{:.3f}'),
                               ('cont_rel', 'continuum [rel]', '{:.2e}'), ('rms_norm', 'RMS(dI/I_c)', '{:.2e}'),
                               ('max_norm', 'max |dI|/I_c', '{:.2e}')):
        g = values(rows, 'graphnet', field, absolute=True)
        l = values(rows, 'lte', field, absolute=True)
        sk = skill(rows, field)
        P(f'   {label:<30s} {fmt(np.median(g), spec):>11s} {fmt(np.median(l), spec):>11s} {fmt(sk["skill"], "{:.3f}"):>7s} '
          f'{100 * sk["win_fraction"]:>8.1f}% {fmt(sk["ratio_p50"], "{:.3f}"):>12s} {fmt(sk["ratio_p90"], "{:.3f}"):>8s}')
    vb = values(rows, 'graphnet', 'v_bias_ms')
    P(f'   GraphNet velocity bias: median {np.median(vb):+.1f} m/s, robust std (1.48 MAD) {1.4826 * np.median(np.abs(vb - np.median(vb))):.1f} m/s, '
      f'|bias| p99 {np.quantile(np.abs(vb), .99):.1f} m/s  (core searched within +-{CORE_SEARCH_NM} nm of the reference core)')
    de = values(rows, 'graphnet', 'depth_err')
    P(f'   GraphNet depth bias   : median {np.median(de):+.5f} I_c, robust std {1.4826 * np.median(np.abs(de - np.median(de))):.5f} I_c, '
      f'mean {np.mean(de):+.5f} I_c')

    # ---- 4. departure coefficients
    P('\n4. DEPARTURE COEFFICIENTS WHERE THE LINE FORMS  (weighted by the line-core contribution function)')
    P(f'   {"quantity":<44s} {"GraphNet":>11s} {"LTE (b=1)":>11s}')
    for field, label in (('cf_mae_lower', '|d log10 b_lower|   (line opacity)      [dex]'),
                         ('cf_mae_upper', '|d log10 b_upper|                        [dex]'),
                         ('cf_mae_ratio', '|d log10 (b_u/b_l)| (line source funct.) [dex]'),
                         ('cf_bias_lower', 'bias log10 b_lower                       [dex]'),
                         ('cf_bias_ratio', 'bias log10 (b_u/b_l)                     [dex]'),
                         ('mae_masked', 'MAE all 16 levels, loss mask (n/N>1e-9) [dex]'),
                         ('mae_all', 'MAE all 16 levels, unmasked              [dex]'),
                         ('pop_cons_cf', '|sum_i n_i / N_Si - 1|')):
        g = values(rows, 'dep_graphnet', field)
        l = values(rows, 'dep_lte', field)
        P(f'   {label:<44s} {fmt(np.median(g), "{:.4f}"):>11s} {fmt(np.median(l), "{:.4f}"):>11s}')
    zs = values(rows, 'formation', 'z_tau1_shift_km')
    P(f'   tau_core = 1 height: reference median {np.nanmedian(values(rows, "formation", "z_tau1_core_km")):.0f} km '
      f'(continuum {np.nanmedian(values(rows, "formation", "z_tau1_cont_km")):.0f} km); '
      f'shift induced by GraphNet populations: median |dz| {np.median(np.abs(zs)):.1f} km, p99 {np.quantile(np.abs(zs), .99):.1f} km')
    P(f'   90% of the core contribution function lies between {np.median(values(rows, "formation", "z_cf_5_km")):.0f} and '
      f'{np.median(values(rows, "formation", "z_cf_95_km")):.0f} km (medians of the per-column 5-95% bounds)')

    # ---- 5. stratification
    P('\n5. BY COLUMN ORIGIN  (median RMS(dI/I_c); fraction below the largest noise level)')
    s_min = max(noise_levels)
    P(f'   {"origin":<16s} {"n":>5s} {"GraphNet":>10s} {"LTE":>10s} {"floor":>10s} {"skill":>7s} '
      f'{"GNN<sigma":>10s} {"|dv| m/s":>9s} {"|d depth|":>10s}')
    for origin in ('bifrost', 'semi-empirical'):
        sub = [r for r in rows if r['origin'] == origin]
        if not sub:
            continue
        g = values(sub, 'graphnet', 'rms_norm')
        sk = skill(sub, 'rms_norm')
        P(f'   {origin:<16s} {len(sub):>5d} {fmt(np.median(g)):>10s} {fmt(np.median(values(sub, "lte", "rms_norm"))):>10s} '
          f'{fmt(np.median(values(sub, "floor", "rms_norm"))):>10s} {fmt(sk["skill"], "{:.3f}"):>7s} {100 * np.mean(g < s_min):>9.1f}% '
          f'{np.median(values(sub, "graphnet", "v_bias_ms", absolute=True)):>9.1f} '
          f'{np.median(values(sub, "graphnet", "depth_err", absolute=True)):>10.4f}')
    # error vs line strength
    d = np.array([r['reference']['depth'] for r in rows])
    g = np.array([r['graphnet']['rms_norm'] for r in rows])
    P('   by reference line depth:')
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum():
            P(f'      depth {lo:.1f}-{min(hi, 1.0):.1f}: n = {m.sum():4d}, GraphNet RMS median {np.median(g[m]):.2e}, '
              f'p99 {np.quantile(g[m], .99):.2e}, < {s_min:g}: {100 * np.mean(g[m] < s_min):.1f}%')

    # ---- 6. worst columns
    P('\n6. WORST COLUMNS by deployed RMS(dI/I_c)  (index into the split, for inspection)')
    order = np.argsort(-g)[:8]
    P(f'   {"index":>7s} {"origin":<15s} {"RMS":>9s} {"max":>9s} {"d depth":>9s} {"d peak":>8s} {"dv m/s":>8s} '
      f'{"depth_ref":>10s} {"peak_ref":>9s} {"z_tau1 km":>10s} {"LTE RMS":>9s}')
    for k in order:
        r = rows[k]
        P(f'   {r["index"]:>7d} {r["origin"]:<15s} {r["graphnet"]["rms_norm"]:>9.2e} {r["graphnet"]["max_norm"]:>9.2e} '
          f'{r["graphnet"]["depth_err"]:>+9.4f} {r["graphnet"]["peak_err"]:>+8.3f} {fmt(r["graphnet"]["v_bias_ms"], "{:+.1f}"):>8s} '
          f'{r["reference"]["depth"]:>10.3f} {r["reference"]["peak"]:>9.3f} '
          f'{r["formation"]["z_tau1_core_km"]:>10.0f} {r["lte"]["rms_norm"]:>9.2e}')
    P('   (peak = max(I)/I_c - 1: an emission core shows as peak_ref > 0 with depth_ref ~ 0)')

    # ---- 7. timing
    P('\n7. ACCELERATION  (wall-clock per column, medians)')
    tb = np.median([r['timing']['t_build'] for r in rows])
    ti = np.median([r['timing']['t_iterate'] for r in rows])
    tf = np.median([r['timing']['t_formal'] for r in rows])
    P(f'   atmosphere + LTE setup (shared by both paths)  : {tb * 1e3:8.1f} ms')
    P(f'   NLTE iteration to convergence (replaced)       : {ti * 1e3:8.1f} ms')
    P(f'   single formal solution (shared)                : {tf * 1e3:8.1f} ms')
    if timing:
        tg_b, tg_s = timing['gnn_per_column_batched_s'], timing['gnn_single_column_s']
        P(f'   GraphNet, batch of {timing["gnn_batch_size"]} on {timing["gnn_device"]:<8s}            : {tg_b * 1e3:8.1f} ms per column')
        P(f'   GraphNet, one column on {timing["gnn_device"]:<8s}                : {tg_s * 1e3:8.1f} ms')
        P(f'   speed-up of the NLTE step   : {ti / tg_b:6.1f}x batched, {ti / tg_s:6.1f}x single column')
        P(f'   speed-up end-to-end         : {(tb + ti + tf) / (tb + tg_b + tf):6.2f}x batched, '
          f'{(tb + ti + tf) / (tb + tg_s + tf):6.2f}x single (setup + formal solution dominate)')
    else:
        P('   GraphNet timing not measured (predictions read from a pickle; use --ck to time the network)')

    # ---- 8. consistency
    P('\n8. PIPELINE CONSISTENCY')
    dc = values(rows, 'consistency', 'dep_conv_vs_stored_max')
    P(f'   converged solve here vs stored database departure coefficients: max |d log10 b| median {np.median(dc):.1e}, '
      f'worst {np.max(dc):.1e}  ({sum(not r["converged"] for r in rows)} columns failed to converge)')
    if rows and 'I_conv_vs_stored_median' in rows[0]['consistency']:
        ic = values(rows, 'consistency', 'I_conv_vs_stored_median')
        im = values(rows, 'consistency', 'I_conv_vs_stored_max')
        P(f'   converged solve here vs stored emergent intensity (database grid): |dI/I| median {np.median(ic):.1e}, '
          f'worst column max {np.max(im):.1e}')
    P(f'   formal-solution floor (no SE iteration) vs converged: median RMS {np.median(values(rows, "floor", "rms_norm")):.1e} I_c, '
      f'shape-only {np.median(values(rows, "floor", "rms_shape")):.1e} I_c -- the irreducible error of the deployed path')
    return '\n'.join(L)


def headline_summary(rows, noise_levels, timing, checkpoint, source, args):
    """Flat dict of the numbers worth tracking from checkpoint to checkpoint."""
    s = {'checkpoint': checkpoint, 'source': source, 'split': args.dtst, 'datadir': args.rd,
         'n_columns': len(rows), 'timing': timing or {}}
    for name in TRIALS:
        for field in ('rms_norm', 'rms_shape', 'max_norm', 'core_norm', 'rel_median'):
            s[f'{name}.{field}'] = tail(values(rows, name, field))
        for field in ('depth_err', 'ew_rel', 'v_bias_ms', 'fwhm_err_pm', 'cont_rel'):
            s[f'{name}.abs_{field}'] = tail(values(rows, name, field, absolute=True))
        for lev in noise_levels:
            s[f'{name}.frac_rms_below_{lev:g}'] = float(np.mean(values(rows, name, 'rms_norm') < lev))
            s[f'{name}.frac_chi2nu_le1_{lev:g}'] = float(np.mean(values(rows, name, f'chi2nu_{lev:g}') <= 1))
    for field in ('rms_norm', 'depth_err', 'ew_rel', 'v_bias_ms', 'fwhm_err_pm'):
        s[f'skill.{field}'] = skill(rows, field)
    for field in ('cf_mae_lower', 'cf_mae_upper', 'cf_mae_ratio', 'mae_masked', 'pop_cons_cf'):
        s[f'dep_graphnet.{field}'] = tail(values(rows, 'dep_graphnet', field))
        s[f'dep_lte.{field}'] = tail(values(rows, 'dep_lte', field))
    s['formation.abs_z_tau1_shift_km'] = tail(values(rows, 'formation', 'z_tau1_shift_km', absolute=True))
    for origin in ('bifrost', 'semi-empirical'):
        sub = [r for r in rows if r['origin'] == origin]
        if sub:
            s[f'{origin}.graphnet.rms_norm'] = tail(values(sub, 'graphnet', 'rms_norm'))
            s[f'{origin}.lte.rms_norm'] = tail(values(sub, 'lte', 'rms_norm'))
    return s


# ----------------------------------------------------------------------------- figures
def _style(ax):
    ax.grid(True, lw=0.5, alpha=0.25, color=COLOR['grid'])
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _cdf(ax, x, color, label, ls='-'):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    if x.size:
        ax.step(x, np.arange(1, x.size + 1) / x.size, where='post', color=color, lw=2.0, label=label, ls=ls)


def fig_profiles(rows, residuals, profiles, wave, noise, path, plt):
    """Three columns (best, median, worst by deployed RMS): normalised profiles and residuals."""
    rms = np.array([r['graphnet']['rms_norm'] for r in rows])
    order = np.argsort(rms)
    picks = [(order[0], 'best'), (order[len(order) // 2], 'median'), (order[-1], 'worst')]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for c, (k, tag) in enumerate(picks):
        r, p = rows[k], profiles[k]
        Ic = continuum_level(p['I_conv'])
        a = ax[0, c]
        _style(a)
        a.plot(wave, p['I_conv'] / Ic, lw=4.5, color=COLOR['reference'], alpha=0.30, solid_capstyle='round',
               label='converged NLTE (truth)')
        a.plot(wave, p['I_gnn'] / Ic, lw=1.8, color=COLOR['graphnet'], label='GraphNet + formal solution')
        a.plot(wave, p['I_lte'] / Ic, lw=1.8, color=COLOR['lte'], ls=(0, (5, 2)), label='LTE')
        a.set_title(f'{tag}: column {r["index"]} ({r["origin"]}), depth {r["reference"]["depth"]:.2f}, '
                    f'RMS {r["graphnet"]["rms_norm"]:.1e}', fontsize=10, loc='left')
        if c == 0:
            a.set_ylabel(r'$I/I_c$')
            a.legend(frameon=False, fontsize=8.5, loc='lower left')
        b = ax[1, c]
        _style(b)
        b.axhspan(-noise * 1e3, noise * 1e3, color=COLOR['grid'], alpha=0.18, lw=0)
        b.axhline(0, color=COLOR['reference'], lw=0.8)
        b.plot(wave, residuals[k]['lte'] * 1e3, lw=1.6, color=COLOR['lte'], ls=(0, (5, 2)), label='LTE')
        b.plot(wave, residuals[k]['floor'] * 1e3, lw=1.6, color=COLOR['floor'], label='formal floor')
        b.plot(wave, residuals[k]['graphnet'] * 1e3, lw=1.8, color=COLOR['graphnet'], label='GraphNet')
        lim = max(3 * noise * 1e3, 1.2 * np.max(np.abs(residuals[k]['graphnet'])) * 1e3)
        b.set_ylim(-lim, lim)
        b.set_xlabel('wavelength [nm]')
        if c == 0:
            b.set_ylabel(r'$(I - I_{\rm ref})/I_c$  [$10^{-3}$]')
            b.legend(frameon=False, fontsize=8.5, loc='lower left', ncol=3)
        b.annotate(f'grey band: $\\pm{noise:g}\\,I_c$ noise', xy=(0.99, 0.92), xycoords='axes fraction',
                   ha='right', fontsize=8, color=COLOR['muted'])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_error_spectrum(residuals, wave, noise_levels, path, plt):
    """Median and 99th percentile of |dI|/I_c at every wavelength, over columns."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    _style(ax)
    for name, ls in (('lte', (0, (5, 2))), ('floor', '-'), ('graphnet', '-')):
        R = np.abs(np.stack([res[name] for res in residuals]))
        med, p99 = np.median(R, axis=0), np.quantile(R, 0.99, axis=0)
        ax.plot(wave, med, lw=2.0, color=COLOR[name], ls=ls, label=f'{TRIAL_LABEL[name]}, median')
        ax.fill_between(wave, med, p99, color=COLOR[name], alpha=0.15, lw=0)
    for s in noise_levels:
        ax.axhline(s, color=COLOR['muted'], lw=0.9, ls=':')
        ax.annotate(f'noise {s:g}', xy=(wave[-3], s * 1.15), fontsize=8, color=COLOR['muted'], ha='right')
    ax.axvline(LINE_CORE_NM, color=COLOR['grid'], lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel(r'$|I - I_{\rm conv}|\,/\,I_c$')
    ax.set_title('Error spectrum over columns: line = median, band = median to 99th percentile', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_distributions(rows, noise_levels, path, plt):
    """CDF of the per-column RMS and the paired GraphNet-vs-LTE scatter."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    _style(a)
    for name, ls in (('lte', (0, (5, 2))), ('floor', '-'), ('network', (0, (1, 1))), ('graphnet', '-')):
        _cdf(a, values(rows, name, 'rms_norm'), COLOR[name], TRIAL_LABEL[name], ls)
    for s in noise_levels:
        a.axvline(s, color=COLOR['muted'], lw=0.9, ls=':')
        a.annotate(f'{s:g}', xy=(s, 0.02), fontsize=8, color=COLOR['muted'], ha='right', rotation=90)
    a.set_xscale('log')
    a.set_xlabel(r'RMS$(I - I_{\rm ref})/I_c$ per column')
    a.set_ylabel('fraction of columns')
    a.set_ylim(0, 1)
    a.set_title('Cumulative distribution; dotted verticals are the noise levels', fontsize=10, loc='left')
    a.legend(frameon=False, fontsize=8.5, loc='upper left')

    b = ax[1]
    _style(b)
    x = np.array([r['lte']['rms_norm'] for r in rows])
    y = np.array([r['graphnet']['rms_norm'] for r in rows])
    lim = [max(1e-7, min(x.min(), y.min()) * 0.5), max(x.max(), y.max()) * 2]
    b.plot(lim, lim, lw=1.0, color=COLOR['grid'])
    for origin, marker in (('bifrost', 'o'), ('semi-empirical', 's')):
        m = np.array([r['origin'] == origin for r in rows])
        if m.any():
            b.scatter(x[m], y[m], s=22, marker=marker, color=COLOR[origin], alpha=0.7, edgecolor='white',
                      linewidth=0.5, label=f'{origin} ({m.sum()})')
    b.set_xscale('log')
    b.set_yscale('log')
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_xlabel(r'LTE RMS$(dI/I_c)$')
    b.set_ylabel(r'GraphNet RMS$(dI/I_c)$')
    won = 100 * np.mean(y < x)
    b.set_title(f'Paired per column: GraphNet better on {won:.1f}% (below the diagonal)', fontsize=10, loc='left')
    b.legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_line_parameters(rows, path, plt):
    """CDFs of |error| for the four inversion-relevant line parameters, GraphNet vs LTE."""
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for a, (field, label) in zip(ax, (('depth_err', r'$|\Delta$ line depth$|$  [$I_c$]'),
                                      ('ew_rel', r'$|\Delta W|/W$  (equivalent width)'),
                                      ('v_bias_ms', r'$|\Delta v_{\rm core}|$  [m s$^{-1}$]'),
                                      ('fwhm_err_pm', r'$|\Delta$ FWHM$|$  [pm]'))):
        _style(a)
        _cdf(a, values(rows, 'lte', field, absolute=True), COLOR['lte'], 'LTE', (0, (5, 2)))
        _cdf(a, values(rows, 'floor', field, absolute=True), COLOR['floor'], 'formal floor')
        _cdf(a, values(rows, 'graphnet', field, absolute=True), COLOR['graphnet'], 'GraphNet')
        a.set_xscale('log')
        a.set_ylim(0, 1)
        a.set_xlabel(label)
        sk = skill(rows, field)
        a.set_title(f'skill {sk["skill"]:.2f}, GraphNet wins {100 * sk["win_fraction"]:.0f}%', fontsize=10, loc='left')
    ax[0].set_ylabel('fraction of columns')
    ax[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_departure(rows, profiles, path, plt):
    """Departure-coefficient errors of the two line levels against height, next to where the line forms."""
    z_edges = np.arange(-200, 2600, 100.0)
    z_mid = 0.5 * (z_edges[1:] + z_edges[:-1])

    def binned(key_gnn, key_ref):
        acc = [[] for _ in z_mid]
        for p in profiles:
            zk = p['z'] / 1e3
            d = np.abs(p[key_gnn] - p[key_ref])
            k = np.digitize(zk, z_edges) - 1
            ok = (k >= 0) & (k < len(z_mid))
            for kk, dd in zip(k[ok], d[ok]):
                acc[kk].append(dd)
        med = np.array([np.median(a) if a else np.nan for a in acc])
        p90 = np.array([np.quantile(a, 0.9) if a else np.nan for a in acc])
        return med, p90

    def binned_lte(key_ref):
        acc = [[] for _ in z_mid]
        for p in profiles:
            zk = p['z'] / 1e3
            k = np.digitize(zk, z_edges) - 1
            ok = (k >= 0) & (k < len(z_mid))
            for kk, dd in zip(k[ok], np.abs(p[key_ref])[ok]):
                acc[kk].append(dd)
        return np.array([np.median(a) if a else np.nan for a in acc])

    # mean normalised contribution function on the same height bins
    cf = np.zeros(len(z_mid))
    for p in profiles:
        zk = p['z'] / 1e3
        k = np.digitize(zk, z_edges) - 1
        ok = (k >= 0) & (k < len(z_mid))
        np.add.at(cf, k[ok], p['w_cf'][ok])
    cf /= max(cf.sum(), 1e-30)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1, 1, 0.55]})
    for a, (kg, kr, title) in zip(ax[:2], (('dep_gnn_lower', 'dep_ref_lower', r'lower level, $\log_{10} b_l$ (line opacity)'),
                                          ('dep_gnn_ratio', 'dep_ref_ratio', r'$\log_{10}(b_u/b_l)$ (line source function)'))):
        _style(a)
        med, p90 = binned(kg, kr)
        a.plot(z_mid, binned_lte(kr), lw=2.0, color=COLOR['lte'], ls=(0, (5, 2)), label='LTE (error = |NLTE effect|)')
        a.plot(z_mid, med, lw=2.0, color=COLOR['graphnet'], label='GraphNet, median')
        a.fill_between(z_mid, med, p90, color=COLOR['graphnet'], alpha=0.18, lw=0, label='GraphNet, median to p90')
        a.set_yscale('log')
        a.set_xlabel('height [km]')
        a.set_ylabel('|error| [dex]')
        a.set_title(title, fontsize=10, loc='left')
        a.legend(frameon=False, fontsize=8.5, loc='upper left')
        zt = np.nanmedian([r['formation']['z_tau1_core_km'] for r in rows])
        a.axvline(zt, color=COLOR['reference'], lw=0.9, ls=':')
        a.annotate(r'$\tau_{\rm core}=1$', xy=(zt, a.get_ylim()[0] * 1.5), fontsize=8, color=COLOR['muted'])
    a = ax[2]
    _style(a)
    a.fill_between(z_mid, 0, cf, color=COLOR['reference'], alpha=0.35, lw=0)
    a.set_xlabel('height [km]')
    a.set_ylabel('mean line-core contribution (normalised)')
    a.set_title('where the 1083 nm core forms', fontsize=10, loc='left')
    a.set_xlim(ax[0].get_xlim())
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_stratified(rows, noise_levels, path, plt):
    """Deployed error against line strength and formation height, marked by column origin."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    y = np.array([r['graphnet']['rms_norm'] for r in rows])
    for a, (xs, xlabel) in zip(ax, ((np.array([r['reference']['depth'] for r in rows]), 'reference line depth  1 - I_core/I_c'),
                                    (np.array([r['formation']['z_tau1_core_km'] for r in rows]), r'reference $\tau_{\rm core}=1$ height [km]'))):
        _style(a)
        for origin, marker in (('bifrost', 'o'), ('semi-empirical', 's')):
            m = np.array([r['origin'] == origin for r in rows])
            if m.any():
                a.scatter(xs[m], y[m], s=20, marker=marker, color=COLOR[origin], alpha=0.65, edgecolor='white',
                          linewidth=0.5, label=f'{origin} ({m.sum()})')
        for s in noise_levels:
            a.axhline(s, color=COLOR['muted'], lw=0.9, ls=':')
        a.set_yscale('log')
        a.set_xlabel(xlabel)
        a.set_ylabel(r'GraphNet RMS$(dI/I_c)$')
    ax[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    ax[0].set_title('Deployed error vs line strength (dotted: noise levels)', fontsize=10, loc='left')
    ax[1].set_title('Deployed error vs formation height', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(rows, residuals, profiles, wave, noise_levels, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = {}
    for tag, fn, argv in (('profiles', fig_profiles, (rows, residuals, profiles, wave, max(noise_levels))),
                          ('error_spectrum', fig_error_spectrum, (residuals, wave, noise_levels)),
                          ('distributions', fig_distributions, (rows, noise_levels)),
                          ('line_parameters', fig_line_parameters, (rows,)),
                          ('departure', fig_departure, (rows, profiles)),
                          ('stratified', fig_stratified, (rows, noise_levels))):
        path = f'{prefix}_{tag}.png'
        fn(*argv, path, plt)
        paths[tag] = path
    return paths


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rd', '--readir', default='../data_1d_si_v3/', metavar='READIR',
                   help='directory holding the database pickles')
    p.add_argument('--dtst', '--dataset', default='validation', metavar='DTST',
                   help="split to evaluate ('validation' is the independent snapshot)")
    p.add_argument('--ck', '--checkpoint', default=None, metavar='CKPT',
                   help='*.pth file or run directory (latest *best.pth in it); the model is run here')
    p.add_argument('--pred', default=None, metavar='PKL',
                   help='prediction pickle from test_prediction.py; its predictions are evaluated as-is')
    p.add_argument('--n', default=500, type=int, metavar='N', help='number of columns to evaluate')
    p.add_argument('--seed', default=0, type=int, metavar='SEED', help='seed for column selection and bootstrap')
    p.add_argument('--device', default='cpu', metavar='DEV', help="torch device for --ck, e.g. 'cpu' or 'cuda:0'")
    p.add_argument('--batch', default=64, type=int, metavar='B', help='GraphNet batch size for --ck')
    p.add_argument('--workers', default=16, type=int, metavar='W', help='parallel lightweaver processes')
    p.add_argument('--mu', default=None, type=float, metavar='MU',
                   help='viewing angle cosine (default: atmos.muz[-1] = 0.9531, as in the database)')
    p.add_argument('--noise', default=list(DEFAULT_NOISE), type=float, nargs='+', metavar='SIGMA',
                   help='photon-noise levels in units of the continuum')
    p.add_argument('--sav', '--savedir', default=None, metavar='SAVEDIR',
                   help='output directory (default: <run dir>/acceptance/)')
    p.add_argument('--no-fig', action='store_true', help='skip the figures')
    args = p.parse_args()

    if args.pred is None and args.ck is None:
        import api
        args.ck = api.DEFAULT_CHECKPOINT
    noise_levels = sorted(set(args.noise), reverse=True)
    rng = np.random.default_rng(args.seed)

    # ---- network output
    timing, hyperparams = None, None
    if args.pred is not None:
        preds_all, P = load_predictions(args.pred)
        checkpoint = str(P.get('checkpoint', 'unknown'))
        hyperparams = dict(P.get('hyperparams', {}) or {})
        source = f'prediction pickle {args.pred}'
        run_dir = os.path.dirname(os.path.abspath(args.pred))
    else:
        import api
        checkpoint = api._resolve_checkpoint(args.ck)
        source = 'model run here through api._build_graph'
        run_dir = os.path.dirname(os.path.abspath(checkpoint))
        preds_all = None
    savedir = args.sav or os.path.join(run_dir, 'acceptance')
    os.makedirs(savedir, exist_ok=True)

    print(f'=> reading {args.dtst} split from {args.rd}')
    data = load_split(args.rd, args.dtst)
    n_total = len(data['T'])
    if preds_all is not None and len(preds_all) != n_total:
        raise ValueError(f'prediction pickle holds {len(preds_all)} columns, the split {n_total}: not the same split/database')
    indices = rng.permutation(n_total)[:min(args.n, n_total)]
    indices = np.array([i for i in indices if np.all(np.isfinite(data['logdeparture'][i]))])

    if preds_all is None:
        print(f'=> running {checkpoint} on {args.device} over {len(indices)} columns')
        predictions, hyperparams, timing = predict_with_model(checkpoint, data, indices, args.device, args.batch)
    else:
        predictions = {int(i): preds_all[int(i)] for i in indices}
        print(f'=> using predictions from {args.pred} (checkpoint {checkpoint})')

    wave = np.linspace(WAVE_LO, WAVE_HI, WAVE_N)
    stored_wave = data['wave']
    stored_mask = None
    if stored_wave is not None:
        stored_mask = (stored_wave >= WAVE_LO) & (stored_wave <= WAVE_HI)

    tasks = []
    for i in indices:
        i = int(i)
        dep_ref = np.asarray(data['logdeparture'][i], dtype=np.float64)
        mask = None
        if data['n_Nat'] is not None:
            with np.errstate(invalid='ignore'):
                mask = np.asarray(data['n_Nat'][i]) >= NEGLIGIBLE_LOG_N_OVER_NTOT
        tasks.append({'index': i, 'atmos': column_arrays(data, i), 'dep_ref': dep_ref,
                      'dep_gnn': np.asarray(predictions[i], dtype=np.float64), 'mask': mask, 'mu': args.mu,
                      'stored_wave': np.ascontiguousarray(stored_wave[stored_mask]) if stored_wave is not None else None,
                      'stored_Iwave': np.asarray(data['Iwave'][i])[stored_mask] if data['Iwave'] is not None else None})
    by_index = {t['index']: t for t in tasks}
    n_vturb_neg = sum(np.any(t['atmos'][3] < 0) for t in tasks)

    # ---- synthesis
    from tqdm import tqdm
    rows, residuals, profiles, failures = [], [], [], []
    t_start = time.time()
    workers = max(1, min(args.workers, len(tasks)))
    print(f'=> synthesising 4 profiles per column (converged NLTE + 3 deployed-path syntheses) on {workers} processes')
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(var, '1')       # one thread per lightweaver process
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, initializer=_worker_init, initargs=(wave,)) as pool:
        for res in tqdm(pool.imap_unordered(synthesise_column, tasks, chunksize=1), total=len(tasks), ncols=100):
            if not res['ok']:
                failures.append((res['index'], res['error']))
                continue
            row, resid, prof = column_metrics(res, by_index[res['index']], wave, noise_levels)
            rows.append(row)
            residuals.append(resid)
            profiles.append(prof)
    wall = time.time() - t_start
    if failures:
        print(f'   {len(failures)} columns failed in lightweaver:')
        for i, e in failures[:10]:
            print(f'     column {i}: {e}')
    if not rows:
        print('No columns evaluated.')
        return
    # deterministic order (imap_unordered)
    order = np.argsort([r['index'] for r in rows])
    rows = [rows[k] for k in order]
    residuals = [residuals[k] for k in order]
    profiles = [profiles[k] for k in order]

    # ---- report
    report = build_report(rows, args, checkpoint, source, timing, hyperparams, noise_levels, rng)
    if n_vturb_neg:
        report += (f'\n\n  NOTE: {n_vturb_neg} columns carry vturb < 0 (clipped to 0 for the synthesis, fed raw to the '
                   f'network as in training); api.compute_dep_coeffs would reject them.')
    report += f'\n\n  {len(rows)} columns in {wall / 60:.1f} min wall-clock on {workers} processes.'
    print('\n' + report)

    stamp = time.strftime('%Y%m%d-%H%M%S')
    prefix = os.path.join(savedir, f'acceptance_{args.dtst}_{stamp}')
    with open(prefix + '.txt', 'w') as fh:
        fh.write(report + '\n')
    summary = headline_summary(rows, noise_levels, timing, checkpoint, source, args)
    with open(prefix + '.json', 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)
    with open(prefix + '.pkl', 'wb') as fh:
        pickle.dump({'rows': rows, 'residuals': residuals, 'profiles': profiles, 'wave': wave,
                     'checkpoint': checkpoint, 'source': source, 'datadir': args.rd, 'split': args.dtst,
                     'hyperparams': hyperparams, 'timing': timing, 'noise_levels': noise_levels,
                     'failures': failures, 'args': vars(args)}, fh)
    written = [prefix + '.txt', prefix + '.json', prefix + '.pkl']
    if not args.no_fig:
        written += list(make_figures(rows, residuals, profiles, wave, noise_levels, prefix).values())
    print('\n  wrote:\n    ' + '\n    '.join(written) + '\n')


if __name__ == '__main__':
    main()
