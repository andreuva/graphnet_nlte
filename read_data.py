from read_header import read_generic_header
from read_nodes import get_data
from configs import PMD
import numpy as np
import os, sys
import copy
import matplotlib.pyplot as plt

# Copy the configuration dictionary
data = copy.deepcopy(PMD)
# Read the header
data['head'] = read_generic_header()
# Read the node data
data['nodes'] = get_data(data)

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
                    pass
        plt.tight_layout()
        plt.savefig(f'../3d_playground/{PMD["vars"][var]}.png')
        plt.show()
        plt.close()
    else:
        print(f'{PMD["vars"][var]} is not read')

# Compute the populations from the density matrix variables
data['nodes']['b'] = np.empty((data['head']['nodes'][2],
                               data['head']['nodes'][1],
                               data['head']['nodes'][0],
                               6))

for zz, plane in enumerate(data['nodes']['dm[N_DM==20]']):
    for yy, column in enumerate(plane):
        for xx, node in enumerate(column):
            pass
