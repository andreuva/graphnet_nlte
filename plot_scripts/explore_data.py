import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


datadir = '../data/splited/'

cmass = pkl.load(open(datadir + 'train_cmass.pkl', 'rb'))
vturb = pkl.load(open(datadir + 'train_vturb.pkl', 'rb'))
temp = pkl.load(open(datadir + 'train_T.pkl', 'rb'))
tau = pkl.load(open(datadir + 'train_tau.pkl', 'rb'))
log_b = pkl.load(open(datadir + 'train_logdeparture.pkl', 'rb'))

Nsamples = len(temp)
Npoints = 100

for i in np.random.randint(0, Nsamples, 20):
    plt.plot(tau[i], temp[i])
    # plt.plot(cmass[i], temp[i])

plt.title('20 profiles from the dataset')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$T$')
# plt.xlabel('Column Mass');  plt.ylabel(r'$T$')
plt.xlim([1e3, 1e-7])
plt.show()

min_tau = np.max([tau[i][0] for i in range(Nsamples)])
max_tau = np.min([tau[i][-1] for i in range(Nsamples)])
tau_homogen = np.geomspace(min_tau, max_tau, Npoints)

temp_homogen = np.zeros((Nsamples, Npoints))
for i in range(Nsamples):
    temp_homogen[i] = np.interp(tau_homogen, tau[i], temp[i])


temp_homogen = np.array(temp_homogen)

plt.plot(tau_homogen, np.median(temp_homogen, axis=0))
plt.fill_between(tau_homogen, np.median(temp_homogen, axis=0) + np.std(temp_homogen, axis=0),
                 np.median(temp_homogen, axis=0) - np.std(temp_homogen, axis=0), alpha=0.2)

plt.title('Mean profile')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$T$')
plt.xlim([1e-1, 1e-5])
plt.show()
