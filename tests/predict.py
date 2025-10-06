from Formal import Formal as graphnet
import numpy as np
import lightweaver as lw
import matplotlib.pyplot as plt
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, \
    Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom


def synth_spectrum(atmos, depthData=False, Nthreads=4, conserveCharge=False, prd=False):
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                            Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca', 'Si')
    spect = aSet.compute_wavelength_grid()
    eqPops = aSet.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=Nthreads, conserveCharge=conserveCharge)

    if depthData:
        ctx.depthData.fill = True

    lw.iterate_ctx_se(ctx, prd=prd, quiet=True)
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    ctx.formal_sol_gamma_matrices()
    if prd:
        ctx.prd_redistribute()
    return ctx


atmosRef = Falc82()
_, atmosRef = lw.multi.read_multi_atmos('../data/data/models_atmos/FALC_82.atmos')

atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.ColumnMass,
                              depthScale=atmosRef.cmass,
                              temperature=atmosRef.temperature,
                              vlos=atmosRef.vlos*0,
                              vturb=atmosRef.vturb,
                              verbose=False)
ctx = synth_spectrum(atmos, depthData=True, conserveCharge=False)

# Compute the departure coefficients
n_active = len(ctx.activeAtoms)
log_departure = np.log10(ctx.activeAtoms[-1].n / ctx.activeAtoms[-1].nStar)
for at in range(1, n_active):
    log_departure = np.append(log_departure,
                            np.log10(ctx.activeAtoms[at].n / ctx.activeAtoms[at].nStar),
                            axis=0)

tau = [atmos.tauRef]
vturb = [atmos.vturb]
vlos = [atmos.vlos]
ne = [atmos.ne]
tt = [atmos.temperature]

model = graphnet()
prediction = model.predict(TT=tt, tau=tau, vturb=vturb, vlos=vlos, ne=ne, readir='checkpoints/1D_multiatom/')

plt.figure(figsize=(15, 10), dpi=200)
for i in range(model.hyperparameters['output_size']):
    plt.plot(prediction[0][:, i], color=f'C{i}', label=f'predicted {i}')

for i in range(log_departure.shape[0]):
    plt.plot(np.moveaxis(log_departure, 0, -1)[:, i], '--', color=f'C{i}', label=f'computed {i}')

plt.xlabel('node')
plt.ylabel('log_10(n/n*)')
plt.title('departure coefficients computed and predicted')
plt.legend(fontsize=3)
plt.show()
