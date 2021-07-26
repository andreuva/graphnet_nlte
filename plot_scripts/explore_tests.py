import pickle
import os
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Dataset import Dataset as dtst
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, \
    Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom


test = []

files = sorted(glob('../test/*/*.pkl'))
type_dtst = ['validation', 'test']
dirs = [os.path.split(files[i])[0] for i in range(len(files))]
plotdirs = [dirs[i] + '/plots/' for i in range(len(files))]
names = [os.path.split(files[i])[1] for i in range(len(files))]
nsamples = np.zeros(len(files))
loss = np.zeros(len(files))
msn = np.zeros(len(files))
latdim = np.zeros(len(files))
nhiden = np.zeros(len(files))
hiden_size = np.zeros(len(files))

for j, file in enumerate(files):
    with open(glob(file)[0], 'rb') as filehandle:
        test.append(pickle.load(filehandle))

    print(file, 'loss: ', test[j]['loss'].mean())
    loss[j] = test[j]['loss'].mean()
    msn[j] = test[j]['hyperparams']['n_message_passing_steps']
    latdim[j] = test[j]['hyperparams']['latent_size']
    nsamples[j] = len(test[j]['target'])
    nhiden[j] = test[j]['hyperparams']['mlp_n_hidden_layers']
    hiden_size[j] = test[j]['hyperparams']['mlp_hidden_size']

    if not os.path.exists(plotdirs[j]):
        os.makedirs(plotdirs[j])

# plt.figure(figsize=(10, 10), dpi=180)
# plt.scatter(msn, loss, s=list(latdim**2/8), alpha=0.5)
# plt.xlabel('Number of message passsing steps')
# plt.ylabel('loss (MSE)')
# plt.title('loss vs msn with size as function of latent dimension')
# plt.savefig('msn_loss.png')
# plt.show()
# plt.close()

# plt.figure(figsize=(10, 10), dpi=180)
# plt.scatter(msn, loss, s=list(latdim**2/8), alpha=0.5)
# plt.xlabel('Number of message passsing steps')
# plt.ylabel('loss (MSE)')
# plt.title('loss vs msn with size as function of latent dimension')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('log_msn_loss.png')
# plt.show()
# plt.close()
# exit()

for j, file in enumerate(files):
    print('loading dataset')
    test_dataset = dtst(test[j]['hyperparams'], test[j]['datadir'], type_dtst[j])
    wave = np.linspace(853.9444, 854.9444, 1001)
    Iwave_lte, Iwave_comp, Iwave_targ = [], [], []
    lte_pops = []

    sampler = np.random.randint(0, nsamples[j], 25)
    print('Computing Intensities based on test the populations')
    for i, indx in enumerate(tqdm(sampler)):

        cmass, tau, vturb, vlos, temp_origin, u, log_dep = test_dataset(indx)
        if test[j]['hyperparams']['node_input_size'] > 1:
            temp = 10**test[j]['features'][indx][:, 0]
        else:
            temp = 10**test[j]['features'][indx]
        temp = temp.reshape(temp_origin.shape)

        log_dep_comp = np.moveaxis(test[j]['prediction'][indx], 0, -1)
        log_dep_true = np.moveaxis(test[j]['target'][indx], 0, -1)

        cmass = cmass.astype('float64')
        vturb = vturb.astype('float64')
        temp = temp.astype('float64')

        ptop = None
        if u != 0:
            ptop = u

        atmos_pre = lw.Atmosphere.make_1d(scale=lw.ScaleType.ColumnMass, depthScale=cmass, temperature=temp,
                                          vlos=vlos, vturb=vturb, Ptop=ptop, verbose=False)
        atmos_pre.quadrature(5)
        aSet_pre = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                    Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
        aSet_pre.set_active('Ca')
        if log_dep_comp.shape[0] > 6:
            aSet_pre.set_active('H', 'Ca', 'Mg')
        spect_pre = aSet_pre.compute_wavelength_grid()

        eqPops_pre = aSet_pre.compute_eq_pops(atmos_pre)
        ctx_pre = lw.Context(atmos_pre, spect_pre, eqPops_pre, Nthreads=1, conserveCharge=False)

        # Compute the intensity with the LTE populations
        Iwave_lte.append(ctx_pre.compute_rays(wave, [atmos_pre.muz[-1]], stokes=False))

        nstar = eqPops_pre.atomicPops['Ca'].nStar
        if log_dep_comp.shape[0] > 6:
            nstar = eqPops_pre.atomicPops['H'].nStar
            nstar = np.append(nstar, eqPops_pre.atomicPops['Mg'].nStar, axis=0)
            nstar = np.append(nstar, eqPops_pre.atomicPops['Ca'].nStar, axis=0)

        pops_true = 10**log_dep_true * nstar
        pops_comp = 10**log_dep_comp * nstar

        dep_lte = np.log10(eqPops_pre.atomicPops['Ca'].n / eqPops_pre.atomicPops['Ca'].nStar)
        if log_dep_comp.shape[0] > 6:
            dep_lte = np.log10(eqPops_pre.atomicPops['H'].n / eqPops_pre.atomicPops['H'].nStar)
            dep_lte = np.append(dep_lte, np.log10(eqPops_pre.atomicPops['Mg'].n / eqPops_pre.atomicPops['Mg'].nStar), axis=0)
            dep_lte = np.append(dep_lte, np.log10(eqPops_pre.atomicPops['Ca'].n / eqPops_pre.atomicPops['Ca'].nStar), axis=0)

        lte_pops.append(np.moveaxis(dep_lte, 0, -1))

        atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.ColumnMass, depthScale=cmass, temperature=temp, vlos=vlos, vturb=vturb, verbose=False)
        atmos.quadrature(5)
        aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                                Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
        aSet.set_active('Ca')
        if log_dep_comp.shape[0] > 6:
            aSet.set_active('H', 'Ca', 'Mg')
        spect = aSet.compute_wavelength_grid()

        # Compute the intensity with the target populations
        eqPops = aSet.compute_eq_pops(atmos)
        if log_dep_comp.shape[0] > 6:
            eqPops.atomicPops['H'].n = pops_true[:6, :]
            eqPops.atomicPops['Mg'].n = pops_true[6:-6, :]
            eqPops.atomicPops['Ca'].n = pops_true[-6:, :]
        else:
            eqPops.atomicPops['Ca'].n = pops_true
        ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
        Iwave_targ.append(ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False))

        # Compute the intensity with the output populations
        if log_dep_comp.shape[0] > 6:
            eqPops.atomicPops['H'].n = pops_comp[:6, :]
            eqPops.atomicPops['Mg'].n = pops_comp[6:-6, :]
            eqPops.atomicPops['Ca'].n = pops_comp[-6:, :]
        else:
            eqPops.atomicPops['Ca'].n = pops_comp
        ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
        Iwave_comp.append(ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False))

    print(f'PLOTING AND SAVING FIGURE(S) IN : {dirs[j]}/plots/')
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col')

    print('Ploting and saving sampled populations from the test predictions')

    for i, indx in enumerate(sampler):
        ax.flat[i].plot(lte_pops[i], color='C2')
        ax.flat[i].plot(test[j]['target'][indx], color='C1')
        ax.flat[i].plot(test[j]['prediction'][indx], color='C0')
        # ax.flat[i].text(0.05, 0.85, f'l$\epsilon$={self.eps[i]:5.3f}', transform=ax.flat[i].transAxes)
        axins = inset_axes(ax.flat[i], width="40%", height="40%", loc=1)
        axins.plot(10**test[j]['features'][indx])
        axins.set_ylim([3000, 15000])

    fig.supxlabel(r'$\tau$')
    fig.supylabel('J')

    print(f'saving at: {plotdirs[j]}')
    plt.savefig(plotdirs[j] + f'CaII_checkpoint_{names[j]}_at_{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()

    print('Ploting and saving Intiensities from the sampled populations from the test data')
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col')
    for i, indx in enumerate(tqdm(sampler)):
        ax.flat[i].plot(wave, Iwave_lte[i], color='C2')
        ax.flat[i].plot(wave, Iwave_targ[i], color='C1')
        ax.flat[i].plot(wave, Iwave_comp[i], color='C0')

    print(f'saving at: {plotdirs[j]}')
    plt.savefig(plotdirs[j] + f'Intensities_CaII_checkpoint_{names[j]}_at_{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()
