from read_header import read_generic_header
from read_nodes import get_data
from configs import PMD
import numpy as np
import os, sys
import copy
import matplotlib.pyplot as plt

data = copy.deepcopy(PMD)
data['head'] = read_generic_header()
data['nodes'] = get_data(data)


samples = np.linspace(0, data['head']['nodes0'][2]-1, 16, dtype=int)
# plot the temperature map at diferent heights (in a 4x4 grid figure)
for var in range(5):
    fig, ax = plt.subplots(4,4, figsize=(10,10))
    for i in range(4):
        for j in range(4):
            ax[i,j].set_title(f'{PMD["vars"][var]} at {data["nodes"]["z"][samples[i*4+j]]}')
            ax[i,j].set_xlabel('x')
            ax[i,j].set_ylabel('z')
            ax[i,j].set_aspect('equal')
            ax[i,j].set_axis_off()
            ax[i,j].imshow(data['nodes'][PMD['vars'][var]][samples[i*4+j],:,:], origin='lower', extent=[0,1,0,1])
    
    plt.tight_layout()
    plt.savefig(f'../3d_playground/{PMD["vars"][var]}.png')
    plt.show()
    plt.close()
