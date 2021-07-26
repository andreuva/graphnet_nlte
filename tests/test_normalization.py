import pickle
import os
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.io as io
from Dataset import Dataset as dtst
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, \
    Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom

file = sorted(glob('../test/crd_vlos/*.pkl'))[1]
dirs = os.path.split(file)[0]
plotdirs = dirs + '/plots/'
name = os.path.split(file)[1]

print(f"READING BIFROST: ...\n")
zz = io.readsav('../data/models_atmos/snap530_rh.save')
zz = np.float64(zz['z'][::-1]*1e3)


with open(glob(file)[0], 'rb') as filehandle:
    test = pickle.load(filehandle)
print(file, 'loss: ', test['loss'].mean())
nsamples = len(test['target'])
if not os.path.exists(plotdirs):
    os.makedirs(plotdirs)


print('loading dataset')
test_dataset = dtst(test['hyperparams'], test['datadir'], 'validation')
wave = np.linspace(853.9444, 854.9444, 1001)
nlte_norm = []
ntot = []

sampler = np.random.randint(0, nsamples, 1000)
print('Computing Intensities based on test the populations')
for i, indx in enumerate(tqdm(sampler)):

    cmass, tau, vturb, vlos, temp_origin, u, log_dep = test_dataset(indx)
    if test['hyperparams']['node_input_size'] > 1:
        temp = 10**test['features'][indx][:, 0]
    else:
        temp = 10**test['features'][indx]
    temp = temp.reshape(temp_origin.shape)      

    log_dep_comp = np.moveaxis(test['prediction'][indx], 0, -1)
    log_dep_true = np.moveaxis(test['target'][indx], 0, -1)

    cmass = cmass.astype('float64')
    vturb = vturb.astype('float64')
    temp = temp.astype('float64')

    atmos_pre = lw.Atmosphere.make_1d(scale=lw.ScaleType.ColumnMass, depthScale=cmass, temperature=temp,
                                      vlos=vturb*0, vturb=vturb, verbose=False)
    atmos_pre.quadrature(5)
    aSet_pre = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet_pre.set_active('Ca')
    spect_pre = aSet_pre.compute_wavelength_grid()

    eqPops_pre = aSet_pre.compute_eq_pops(atmos_pre)
    ctx_pre = lw.Context(atmos_pre, spect_pre, eqPops_pre, Nthreads=1, conserveCharge=False)

    nstar = eqPops_pre.atomicPops['Ca'].nStar
    nat = eqPops_pre.atomicPops['Ca'].nTotal
    if log_dep_comp.shape[0] > 6:
        nstar = eqPops_pre.atomicPops['H'].nStar
        nstar = np.append(nstar, eqPops_pre.atomicPops['Mg'].nStar, axis=0)
        nstar = np.append(nstar, eqPops_pre.atomicPops['Ca'].nStar, axis=0)

    pops_true = 10**log_dep_true * nstar
    pops_comp = 10**log_dep_comp * nstar

    # compute the normalization of the departure coefficients
    sum_true = np.sum(pops_true, axis=0)
    sum_comp = np.sum(pops_comp, axis=0)
    ntot.append(sum_true)
    nlte_norm.append((sum_true - sum_comp)/sum_true)

nlte_norm = np.array(nlte_norm)
ntot = np.array(ntot)

save_dir = {'n_ntot_div_ntot': nlte_norm, 'ntot': ntot}

with open(plotdirs + '/normalization.pkl', 'wb') as filehandle:
    pickle.dump(save_dir, filehandle)



fig, ax = plt.subplots()
# plt.figure(figsize=(7.5, 5), dpi=100)
y1, y2 = np.percentile(nlte_norm, (4, 96), axis=0)
ax.fill_between(zz/1e3, y1, y2, color='C1', alpha=0.4)
y1, y2 = np.percentile(nlte_norm, (32, 68), axis=0)
ax.fill_between(zz/1e3, y1, y2, color='C0', alpha=0.4)
ax.plot(zz/1e3, np.median(nlte_norm, axis=0), color='C0')
# ax.set_xticks([0, 1e3, 2e3, 3e3], [0, 1000, 2000, 3000])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.0)

ax.set_xlabel('Height [Km]')
ax.set_ylim((-0.15, 0.15))
ax.set_ylabel(r'$\frac{\sum n - n_{total}}{n_{total}}$')

print(f'saving at: {plotdirs}')
plt.savefig(plotdirs + f'normalization_CaII_checkpoint_{name}_at_{time.strftime("%Y%m%d-%H%M%S")}.pdf', bbox_inches='tight')
plt.close()
