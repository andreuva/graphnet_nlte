from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, \
    Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
import numpy as np
from random import shuffle
import scipy.io as io


def synth_spectrum(atmos, depthData=False, Nthreads=1, conserveCharge=False, prd=False):

    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)

    # Configure the set of atomic models to use.
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                            Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])

    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    # aSet.set_active('H', 'Ca')
    aSet.set_active('Ca')

    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # compute the equilibrium populations at the fixed electron density provided in the model
    eqPops = aSet.compute_eq_pops(atmos)

    # Configure the Context which holds the state of the simulation for the
    # backend, and provides the python interface to the backend.
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=Nthreads, conserveCharge=conserveCharge)

    if depthData:
        ctx.depthData.fill = True

    # Iterate the Context to convergence
    iterate_ctx_crd(ctx, prd=prd)

    # Update the background populations based on the converged solution and
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    # compute the final solution for mu=1 on the provided wavelength grid.
    ctx.formal_sol_gamma_matrices(printUpdate=False)
    if prd:
        ctx.prd_redistribute(printUpdate=False)
    return ctx


def iterate_ctx_crd(ctx, prd=False, Nscatter=10, NmaxIter=500):
    '''
    Iterate a Context to convergence.
    '''
    for i in range(NmaxIter):
        # Compute the formal solution
        dJ = ctx.formal_sol_gamma_matrices(printUpdate=False)
        if prd:
            ctx.prd_redistribute(printUpdate=False)
        # Just update J for Nscatter iterations
        if i < Nscatter:
            continue
        # Update the active populations under statistical equilibrium,
        # conserving charge if this option was set on the Context.
        delta = ctx.stat_equil(printUpdate=False)

        # If we are converged in both relative change of J and populations return
        if dJ < 3e-3 and delta < 1e-3:
            return


bifrost = io.readsav('../data/models_atmos/snap385_rh.save')
bifrost['tg'] = np.reshape(bifrost['tg'], (bifrost['tg'].shape[0], -1))
bifrost['vlos'] = np.reshape(bifrost['vlos'], (bifrost['vlos'].shape[0], -1))
bifrost['nel'] = np.reshape(bifrost['nel'], (bifrost['nel'].shape[0], -1))

# Shufle the bifrost columns
index = np.arange(0, bifrost['tg'].shape[-1])
shuffle(index)
bifrost['tg'] = bifrost['tg'][:, index]
bifrost['vlos'] = bifrost['vlos'][:, index]

tau500 = np.float64(bifrost['z'][::-1]*1e3)
temperature = np.float64(bifrost['tg'][:, index[0]][::-1])
vlos = np.float64(bifrost['vlos'][:, index[0]][::-1])
ne = np.float64(bifrost['nel'][:, index[0]][::-1])
vturb = vlos*0

# Set the depth as the tau500 and the depth scale acordingly
depth = tau500
# depth_scale = lw.ScaleType.Tau500
depth_scale = lw.ScaleType.Geometric

atmos = lw.Atmosphere.make_1d(scale=depth_scale, depthScale=depth, temperature=temperature, vlos=vlos, vturb=vturb, verbose=False, ne=ne)
ctx = synth_spectrum(atmos, depthData=True, conserveCharge=False, prd=False)
tau = atmos.tauRef
cmass = atmos.cmass
# Compute the departure coefficients
log_departure = np.log10(ctx.activeAtoms[0].n / ctx.activeAtoms[0].nStar)
n_Nat = np.log10(ctx.activeAtoms[0].n / ctx.activeAtoms[0].nTotal)
