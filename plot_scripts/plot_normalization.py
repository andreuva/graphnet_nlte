import scipy.io as io
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob

with open('../../test/crd/plots/normalization.pkl', 'rb') as filehandle:
    test = pickle.load(filehandle)

file = sorted(glob('../test/ne/*.pkl'))[0]
dirs = os.path.split(file)[0]
plotdirs = dirs + '/plots/'
name = os.path.split(file)[1]

print(f"READING BIFROST: ...\n")
zz = io.readsav('../data/models_atmos/snap530_rh.save')
zz = np.float64(zz['z'][::-1]*1e3)

ntot = test['ntot']
nlte_norm = test['n_ntot_div_ntot']

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
