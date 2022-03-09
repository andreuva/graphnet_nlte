from read_header import read_generic_header
from read_nodes import get_data
from configs import PMD
import numpy as np
import lte
import copy
import matplotlib.pyplot as plt

# Copy the configuration dictionary
data = copy.deepcopy(PMD)
# Read the header
data['head'] = read_generic_header()
# Read the node data
data['nodes'] = get_data(data)

for var, read in enumerate(PMD['read']):
    if read:
        print(f'{PMD["vars"][var]} read')
    else:
        print(f'{PMD["vars"][var]} is not read')


# select the heights to plot as a index of the z coordinate
samples = np.linspace(0, data['head']['nodes0'][2]-1, 16, dtype=int)
# plot the map of the variables at diferent heights (in a 4x4 grid figure)
for var, read in enumerate(PMD['read']):
    if read:
        fig, ax = plt.subplots(4,4, figsize=(10,10))
        for i in range(4):
            for j in range(4):
                ax[i,j].set_title(f'{PMD["vars"][var]} at {data["nodes"]["z"][samples[i*4+j]]}')
                ax[i,j].set_xlabel('x')
                ax[i,j].set_ylabel('z')
                ax[i,j].set_aspect('equal')
                ax[i,j].set_axis_off()
                if var in PMD['scal']:
                    ax[i,j].imshow(data['nodes'][PMD['vars'][var]][samples[i*4+j],:,:], origin='lower', extent=[0,1,0,1])
                elif var in PMD['vec']:
                    ax[i,j].imshow(data['nodes'][PMD['vars'][var]][samples[i*4+j],:,:,2], origin='lower', extent=[0,1,0,1])
                elif PMD['vars'][var] == 'ContOpac[NLINES=5]':
                    ax[i,j].imshow(data['nodes'][PMD['vars'][var]][samples[i*4+j],:,:,0], origin='lower', extent=[0,1,0,1])
                elif PMD['vars'][var] == 'dm[N_DM==20]':
                    ax[i,j].imshow(data['nodes'][PMD['vars'][var]][samples[i*4+j],:,:,-1], origin='lower', extent=[0,1,0,1])
        plt.tight_layout()
        plt.savefig(f'../3d_playground/{PMD["vars"][var]}.png')
        plt.show()
        plt.close()



# Compute the populations from the density matrix variables
if 'dm[N_DM==20]' in data['nodes'].keys():
    cont = 1 - np.sum(data['nodes']['dm[N_DM==20]'], axis=-1, keepdims=True)
    data['nodes']['n'] = np.append(data['nodes']['dm[N_DM==20]'], cont, axis=-1)
    del data['nodes']['dm[N_DM==20]']

if 'V' in data['nodes'].keys():
    data['nodes']['vz'] = data['nodes']['V'][:,:,:,2]
    del data['nodes']['V']

if 'T' in data['nodes'].keys() and 'n' in data['nodes'].keys():
    # Energy diference between the level and the base level of the ion
    chi_ijk = [1, 2, 3, 4, 5]
    # Ionization potential of the ion
    chi_jk = [0, 1]

    # initialization of the variable that will store the populations
    data['nodes']['n_lte'] = np.empty_like(data['nodes']['n'])
    for iz in range(data['head']['nodes0'][2]):
        for iy in range(data['head']['nodes0'][1]):
            for ix in range(data['head']['nodes0'][0]):
                temp = data['nodes']['temp'][iz,iy,ix]
                # Compute the populations for LTE
                data['nodes']['n_lte'][iz,iy,ix,:] = np.exp(-data['nodes']['n'][iz,iy,ix,:]*temp)
