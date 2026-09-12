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
    Si_atom_custom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom


test = []

type_dtst = 'test'
files = sorted(glob(f'/dat/andreuva/gpu/graphnet/graphnet_nlte/checkpoints_si_v2/20260911-231808/{type_dtst}_checkpoint_*.pkl'))
dirs = [os.path.split(files[i])[0] for i in range(len(files))]
plotdirs = [dirs[i] + '/plots/' for i in range(len(files))]
names = [os.path.split(files[i])[1] for i in range(len(files))]
nsamples = np.zeros(len(files))
loss = np.zeros(len(files))
msn = np.zeros(len(files))
latdim = np.zeros(len(files))
nhiden = np.zeros(len(files))
hiden_size = np.zeros(len(files))

import torch

for j, file in enumerate(files):
    with open(file, 'rb') as filehandle:
        test.append(pickle.load(filehandle))

    if 'train_loss' not in test[j] or test[j]['train_loss'] is None:
        pth_name = os.path.basename(test[j]['checkpoint'])
        pth_path = os.path.join(dirs[j], pth_name)
        if os.path.exists(pth_path):
            ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
            test[j]['train_loss'] = ckpt.get('train_loss', None)
            test[j]['valid_loss'] = ckpt.get('valid_loss', None)

    print(file, 'loss: ', test[j]['loss'].mean())
    loss[j] = test[j]['loss'].mean()
    msn[j] = test[j]['hyperparams']['n_message_passing_steps']
    latdim[j] = test[j]['hyperparams']['latent_size']
    nsamples[j] = len(test[j]['target'])
    nhiden[j] = test[j]['hyperparams']['mlp_n_hidden_layers']
    hiden_size[j] = test[j]['hyperparams']['mlp_hidden_size']

    if not os.path.exists(plotdirs[j]):
        os.makedirs(plotdirs[j])

plt.figure(figsize=(10, 10), dpi=180)
plt.scatter(msn, loss, s=list(latdim**2/8), alpha=0.5)
plt.xlabel('Number of message passsing steps')
plt.ylabel('loss (MSE)')
plt.title('loss vs msn with size as function of latent dimension')
# plt.savefig('msn_loss.png')
# plt.show()
plt.close()

plt.figure(figsize=(10, 10), dpi=180)
plt.scatter(msn, loss, s=list(latdim**2/8), alpha=0.5)
plt.xlabel('Number of message passsing steps')
plt.ylabel('loss (MSE)')
plt.title('loss vs msn with size as function of latent dimension')
plt.xscale('log')
plt.yscale('log')
# plt.savefig('log_msn_loss.png')
# plt.show()
plt.close()

for j in range(len(files)):
    if 'train_loss' in test[j] and test[j]['train_loss'] is not None:
        train_loss = test[j]['train_loss']
        valid_loss = test[j].get('valid_loss', None)
        epochs = np.arange(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 6), dpi=180)
        plt.plot(epochs, train_loss, label='Training Loss')
        if valid_loss is not None and len(valid_loss) == len(train_loss):
            plt.plot(epochs, valid_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'Loss vs Epochs ({names[j]})')
        plt.legend()
        plt.grid(True)
        plt.savefig(plotdirs[j] + f'train_loss_vs_epochs_{names[j]}.png')
        # plt.show()
        plt.close()

for j, file in enumerate(files):
    print('loading dataset')
    test_dataset = dtst(test[j]['hyperparams'], test[j]['datadir'], type_dtst)
    wave = np.linspace(1074.0, 1085.0, 1100)
    Iwave_lte, Iwave_comp, Iwave_targ = [], [], []
    lte_pops = []

    sampler = np.random.randint(0, int(nsamples[j]), 25)
    print('Computing Intensities based on test the populations')
    for i, indx in enumerate(tqdm(sampler)):

        temp_origin, zz, ne, vturb, vlos, u, log_dep = test_dataset(indx)
        temp = temp_origin.astype('float64')

        log_dep_comp = np.moveaxis(test[j]['prediction'][indx], 0, -1)
        log_dep_true = np.moveaxis(test[j]['target'][indx], 0, -1)

        zz = zz.astype('float64')
        vturb = vturb.astype('float64')
        ne = ne.astype('float64')

        ptop = None
        if u != 0:
            ptop = u

        atmos_pre = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=zz, temperature=temp,
                                          vlos=vlos, vturb=vturb, ne=ne, Ptop=ptop, verbose=False)
        atmos_pre.quadrature(5)
        aSet_pre = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom_custom(), Al_atom(), CaII_atom(),
                                    Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
        aSet_pre.set_active('Si')
        spect_pre = aSet_pre.compute_wavelength_grid()

        eqPops_pre = aSet_pre.compute_eq_pops(atmos_pre)
        eqPops_pre.update_lte_atoms_Hmin_pops(atmos_pre, quiet=True)
        ctx_pre = lw.Context(atmos_pre, spect_pre, eqPops_pre, Nthreads=1, conserveCharge=False)

        # Compute the intensity with the LTE populations
        Iwave_lte.append(ctx_pre.compute_rays(wave, [atmos_pre.muz[-1]], stokes=False))

        nstar = eqPops_pre.atomicPops['Si'].nStar

        pops_true = 10**(log_dep_true * 5.0) * nstar
        pops_comp = 10**(log_dep_comp * 5.0) * nstar

        dep_lte = np.log10(eqPops_pre.atomicPops['Si'].n / eqPops_pre.atomicPops['Si'].nStar)
        lte_pops.append(np.moveaxis(dep_lte, 0, -1))

        atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=zz, temperature=temp, vlos=vlos, vturb=vturb, ne=ne, verbose=False)
        atmos.quadrature(5)
        aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom_custom(), Al_atom(), CaII_atom(),
                                Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
        aSet.set_active('Si')
        spect = aSet.compute_wavelength_grid()

        # Compute the intensity with the target populations
        eqPops = aSet.compute_eq_pops(atmos)
        eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
        eqPops.atomicPops['Si'].n = pops_true

        ctx = lw.Context(atmos, spect, eqPops, Nthreads=1, conserveCharge=False)
        Iwave_targ.append(ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False))

        # Compute the intensity with the output populations
        eqPops.atomicPops['Si'].n = pops_comp

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
        temp_orig = test_dataset(indx)[0]
        axins.plot(temp_orig)
        axins.set_ylim([3000, 15000])

    fig.supxlabel(r'$z$')
    fig.supylabel('J')

    print(f'saving at: {plotdirs[j]}')
    plt.savefig(plotdirs[j] + f'Si_checkpoint_{names[j]}_at_{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()

    print('Ploting and saving Intiensities from the sampled populations from the test data')
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex='col')
    for i, indx in enumerate(tqdm(sampler)):
        ax.flat[i].plot(wave, Iwave_lte[i], color='C2')
        ax.flat[i].plot(wave, Iwave_targ[i], color='C1')
        ax.flat[i].plot(wave, Iwave_comp[i], color='C0')

    print(f'saving at: {plotdirs[j]}')
    plt.savefig(plotdirs[j] + f'Intensities_SiI_checkpoint_{names[j]}_at_{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()
