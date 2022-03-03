import os, sys, struct
from configs import CONF
import numpy as np
import copy
from tqdm import tqdm
from scipy import interpolate


def get_data(PMD):
    ''' Reads the pmd for the twolevel module given the header
        Returns true if the reading went well and false otherwise.
    '''

    # Check if there is a file at all
    if not os.path.isfile(PMD['head']['name']):
        return False

    # Initialize output
    inpt = {}
    inpt['nodes'] = (PMD['head']['nodes0'][0],\
                    PMD['head']['nodes0'][1], \
                    PMD['head']['nodes0'][2])

    inpt['x'] = PMD['head']['x'][0:PMD['head']['nodes0'][0]]
    inpt['y'] = PMD['head']['y'][0:PMD['head']['nodes0'][1]]
    inpt['z'] = PMD['head']['z'][0:PMD['head']['nodes0'][2]]

    # Read it as a pmd file
    try:
        with open(PMD['head']['name'],'rb') as f:

            f.seek(PMD['head']['size'], 1)
            f.seek(PMD['head']['msize'], 1)

            T = {}
            for ii,read in enumerate(PMD['read']):
                if read:
                    T[ii] = []

            for iz in tqdm(range(inpt['nodes'][2])):
                for iy in range(inpt['nodes'][1]):
                    for ix in range(inpt['nodes'][0]):
                        # Initialize the counter of node size
                        nsize = 0
                        # Read the variables
                        for ii,read in enumerate(PMD['read']):
                            if read:
                                if ii in PMD['scal']:
                                    bytes = f.read(4)
                                    nsize += 4
                                    T[ii].append(struct.unpack(PMD['head']['endian'] + 'f', bytes)[0])
                                elif ii in PMD['vec']:
                                    for i in range(3):
                                        bytes = f.read(4)
                                        nsize += 4
                                        T[ii].append(struct.unpack(PMD['head']['endian'] + 'f', bytes)[0])
                                elif PMD['vars'][ii] == 'ContOpac[NLINES=5]':
                                    for i in range(5):
                                        bytes = f.read(4)
                                        nsize += 4
                                        T[ii].append(struct.unpack(PMD['head']['endian'] + 'f', bytes)[0])
                                elif PMD['vars'][ii] == 'dm[N_DM==20]':
                                    for i in range(20):
                                        bytes = f.read(8)
                                        nsize += 8
                                        T[ii].append(struct.unpack(PMD['head']['endian'] + 'd', bytes)[0])
                            else:
                                if ii in PMD['scal']:
                                    f.seek(4, 1)
                                    nsize += 4
                                elif ii in PMD['vec']:
                                    f.seek(4*3, 1)
                                    nsize += 4*3
                                elif PMD['vars'][ii] == 'ContOpac[NLINES=5]':
                                    f.seek(4*5, 1)
                                    nsize += 4*5
                                elif PMD['vars'][ii] == 'dm[N_DM==20]':
                                    f.seek(8*20, 1)
                                    nsize += 8*20
                                else:
                                    remaining = PMD['head']['gsize'] - nsize
                                    f.seek(remaining, 1)
                                    # print('readed firts node with {}/584 bytes ignored'.format(remaining))
                                    break

            for ii, read in enumerate(PMD['read']):
                if read:
                    if ii in PMD['scal']:
                        inpt[PMD['vars'][ii]] = np.array(T[ii]).reshape([inpt['nodes'][2], \
                                                                        inpt['nodes'][1], \
                                                                        inpt['nodes'][0]])
                    elif ii in PMD['vec']:
                        inpt[PMD['vars'][ii]] = np.array(T[ii]).reshape([inpt['nodes'][2], \
                                                                        inpt['nodes'][1], \
                                                                        inpt['nodes'][0], \
                                                                        3])
                    elif PMD['vars'][ii] == 'ContOpac[NLINES=5]':
                        inpt[PMD['vars'][ii]] = np.array(T[ii]).reshape([inpt['nodes'][2], \
                                                                        inpt['nodes'][1], \
                                                                        inpt['nodes'][0], \
                                                                        5])
                    elif PMD['vars'][ii] == 'dm[N_DM==20]':
                        inpt[PMD['vars'][ii]] = np.array(T[ii]).reshape([inpt['nodes'][2], \
                                                                        inpt['nodes'][1], \
                                                                        inpt['nodes'][0], \
                                                                        20])
        return inpt
    except:
        raise Exception('Error reading file')
