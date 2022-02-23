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
    inpt['nodes'] = (PMD['head']['nodes0'][0], \
                    PMD['head']['nodes0'][1], \
                    PMD['head']['nodes0'][2])

    inpt['x'] = PMD['head']['x'][0:PMD['head']['nodes0'][0]]
    inpt['y'] = PMD['head']['y'][0:PMD['head']['nodes0'][1]]
    inpt['z'] = PMD['head']['z'][0:PMD['head']['nodes0'][2]]

    # Read it as a pmd file
    try:

        with open(PMD['head']['name'],'rb') as f:

            f.seek(PMD['head']['msize'])

            # Identify scalars and vectors
            scal = [0,1,2,3,4]
            vec = [5,6]
            tens = [7,8,9]
            PMD['vars'] = { 0: 'caii_density', \
                            1: 'e_density', \
                            2: 'hi_density', \
                            3: 'temp', \
                            4: 'micro_velocity', \
                            5: 'B', \
                            6: 'V', \
                            7: 'ContOpac[NLINES=5]', \
                            8: 'dm[N_DM==20]', \
                            9: 'jkq[NLINES]'}


            if 'read' not in PMD.keys():
                PMD['read'] = [False]*len(PMD['vars'].keys())

            PMD['read'][3] = True
            # PMD['read'][5] = True
            # PMD['read'][6] = True
            # Create space to read data


            # Create space to read data
            for ii,read in enumerate(PMD['read']):
                if read:
                    if ii in scal:
                        inpt[ii] = np.empty([inpt['nodes'][2], \
                                            inpt['nodes'][1], \
                                            inpt['nodes'][0]])
                    elif ii in vec:
                        inpt[ii] = np.empty([inpt['nodes'][2], \
                                            inpt['nodes'][1], \
                                            inpt['nodes'][0], \
                                            3])


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
                                if ii in scal:
                                    bytes = f.read(4)
                                    nsize += 4
                                    T[ii].append(struct.unpack(PMD['head']['endian'] + 'f', bytes)[0])
                                elif ii in vec:
                                    for i in range(3):
                                        bytes = f.read(4)
                                        nsize += 4
                                        T[ii].append(struct.unpack(PMD['head']['endian'] + 'f', bytes))
                            else:
                                if ii in scal:
                                    f.seek(4, 1)
                                    nsize += 4
                                elif ii in vec:
                                    f.seek(4*3, 1)
                                    nsize += 4*3
                                elif ii in tens:
                                    remaining = PMD['head']['gsize'] - nsize
                                    f.seek(remaining, 1)
                                    # print('readed firts node with {}/584 bytes ignored'.format(remaining))
                                    break

            for ii, read in enumerate(PMD['read']):
                if read:
                    if ii in scal:
                        inpt[ii] = np.array(T[ii]).reshape([inpt['nodes'][2], \
                                                            inpt['nodes'][1], \
                                                            inpt['nodes'][0]])
                    elif ii in vec:
                        inpt[ii] = np.array(T[ii]).reshape([inpt['nodes'][2], \
                                                            inpt['nodes'][1], \
                                                            inpt['nodes'][0], \
                                                            3])

        return inpt
    except:
        raise Exception('Error reading file')
