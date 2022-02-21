from threading import main_thread
import numpy as np
import matplotlib.pyplot as plt
import struct
import os, sys
from pip import main
from scipy import interpolate
import copy

cgs = r'[erg Hz${}^{-1}$ cm${}^{-2}$ s${}^{-1}$ ' + \
      'sr${}^{-1}$]'
VARS = {'figsize' : [(5,4),(5,7),(9,7),(9,7),(8,9),(8,9), \
                     (12,9),(12,9),(12,9)], \
        'axis_dic': {'x' : 'axis_1_name', \
                     'y' : 'axis_2_name', \
                     'l' : 'axis_3_name'}, \
        'axis_ind': {'x' : 0, \
                     'y' : 1}, \
        'axis_idic': {'axis_1_name' : 'x', \
                      'axis_2_name' : 'y', \
                      'axis_3_name' : 'l'}, \
        'taxis2_dic' : {1: '    Vertical axis:', \
                        2: '       Cut axis 1:'} , \
        'taxis3_dic' : {1: '         Cut axis:', \
                        2: '       Cut axis 2:'} , \
        'ptype_list': ['2D','Lines'], \
        'ptype_dic': {1 : '2D', 2 : 'Lines'}, \
        'ptype_idic': {'2D' : 1, 'Lines' : 2}, \
        'SP_list' : ['I','Q/I','U/I','V/I','P','Log I'], \
        'SPL_dic' : {'I' : r'I '+cgs,\
                     'Q/I' : 'Q/I [%]',\
                     'U/I' : 'U/I [%]', \
                     'V/I' : 'V/I [%]', \
                     'P' : 'P [%]', \
                     'Log I' : r'Log I '+cgs, \
                     'Q' : r'Q '+cgs, \
                     'U' : r'U '+cgs, \
                     'V' : r'V '+cgs, \
                     'P*I' : r'P '+cgs}, \
        'aunit_dic' : {'x' : 'Mm', 'y' : 'Mm', 'l' : 'nm', \
                       'I' : 'CGS', \
                       'Q/I' : '%',\
                       'U/I' : '%', \
                       'V/I' : '%', \
                       'P' : '%', \
                       'Log I' : 'CGS' \
                      }, \
        'format_list' : [], \
        'head_size' : {2: 201844}, \
        'endian' : {'Big' : '>', 'Little' : '<'}, \
        'palet_name' : ['RedYellowBlue','SignAdjustable', \
                        'Greys', 'inferno', 'plasma', \
                        'magma', 'Blues', 'BuGn', \
                        'BuPu', 'GnBu', 'Greens', \
                        'viridis', 'Oranges', 'OrRd', \
                        'PuBu', 'PuBuGn', 'PuRd', \
                        'Purples', 'RdPu', 'Reds', \
                        'YlGn', 'YlGnBu', 'YlOrBr', \
                        'YlOrRd', 'afmhot', 'autumn', \
                        'bone', 'cool', 'copper', \
                        'gist_heat', 'gray', 'hot', \
                        'pink', 'spring', 'summer', \
                        'winter', 'BrBG', 'bwr', \
                        'coolwarm', 'PiYG', 'PRGn', \
                        'PuOr', 'RdBu', 'RdGy', \
                        'RdYlBu', 'RdYlGn', \
                        'Spectral', 'seismic', \
                        'Accent', 'Dark2', 'Paired', \
                        'Pastel1', 'Pastel2', 'Set1', \
                        'Set2', 'Set3', 'gist_earth', \
                        'terrain', 'ocean', \
                        'gist_stern', 'brg', 'CMRmap', \
                        'cubehelix', 'gnuplot', \
                        'gnuplot2', 'gist_ncar', \
                        'nipy_spectral', 'jet', \
                        'rainbow', 'gist_rainbow', \
                        'hsv', 'flag', 'prism'], \
         'lcolor_name' : ['black','blue','green','red', \
                          'cyan','magenta','yellow', \
                          'white'], \
         'lcolor_dict' : {'black' : 'k', 'blue' : 'b', \
                          'green' : 'g','red' : 'r', \
                          'cyan' : 'c','magenta' : 'm', \
                          'yellow' : 'y','white' : 'w'}, \
         'llsty_name' : ['solid','dashed','dotted'], \
         'llsty_dict' : {'solid' : '', 'dashed' : '--', \
                        'dotted' : ":"}, \
         'delete' : 'delete' \
        }
    
CONF = {'wfont_size' : 12, \
        'cfont_size' : 8, \
        'endian' : VARS['endian']['Little'], \
        'format' : 'None', \
        'interpol': False, \
        'nodes': {'x': 100, 'y': 100, 'z': 100}, \
        'col_act': '#008000', \
        'col_pas': '#D3D3D3', \
        'b_pad': 7, \
        'delete' : 'delete' \
        }

PMD = {'data' : None, \
        'head' : None, \
        'delete' : 'delete' \
        }


def check_size(head):
    ''' Checks the size of the pmd file given the header data
    '''

    # Check header
    expected = VARS['head_size'][head['pmd_ver']]
    real = head['size']
    if expected != real:
        return False

    expected += head['nodes'][0]* \
                head['nodes'][1]* \
                head['nodes'][2]* \
                head['gsize'] + \
                head['msize']
    real = os.stat(head['name']).st_size

    if expected == real:
        return True
    else:
        msg = "Expected size {0}. Got {1}"
        msg = msg.format(expected,real)
        raise ValueError(msg)

def skip_plane(self, f, skip):
    ''' Skip a plane
    '''

    bytes = f.seek(skip, 1)

    self.pgc += 100./float(PMD['head']['nodes0'][2])
    pgc = int(self.pgc)
    if pgc > self.pgl:
        diff = pgc - self.pgl
        for ii in range(diff):
            if pgc <= 99:
                self.pg.step()
                self.pg.update_idletasks()
        self.pgl += diff

    return {}


def get_plane(self, f):
    ''' Load the data from a plane in the pmd file
    '''

    # Initialize plane
    T = {}
    for ii in range(12):
        T[ii] = []
    T['done'] = False

    # Read plane
    for iy in range(PMD['head']['nodes0'][1]):
        for ix in range(PMD['head']['nodes0'][0]):
            bytes = f.read(8)
            if PMD['read'][0]:
                T[0].append(struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0])
            bytes = f.read(8)
            if PMD['read'][1]:
                T[1].append(struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0])
            bytes = f.read(8)
            if PMD['read'][2]:
                T[2].append(struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0])
            for ii in range(3):
                bytes = f.read(8)
                if PMD['read'][3]:
                    T[3].append(struct.unpack( \
                                CONF['endian'] + 'd', bytes)[0])
            for ii in range(3):
                bytes = f.read(8)
                if PMD['read'][4]:
                    T[4].append(struct.unpack( \
                                CONF['endian'] + 'd', bytes)[0])
            for ii in range(self.NRL*2):
                bytes = f.read(8)
                if PMD['read'][5]:
                    T[5].append(struct.unpack( \
                                CONF['endian'] + 'd', bytes)[0])
            for ii in range(self.NRU*2):
                bytes = f.read(8)
                if PMD['read'][6]:
                    T[6].append(struct.unpack( \
                                CONF['endian'] + 'd', bytes)[0])
            for ii in range(9):
                bytes = f.read(8)
                if PMD['read'][7]:
                    T[7].append(struct.unpack( \
                                CONF['endian'] + 'd', bytes)[0])
            bytes = f.read(8)
            if PMD['read'][8]:
                T[8].append(struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0])
            bytes = f.read(8)
            if PMD['read'][9]:
                T[9].append(struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0])
            bytes = f.read(8)
            if PMD['read'][10]:
                T[10].append(struct.unpack( \
                                CONF['endian'] + 'd', bytes)[0])
            bytes = f.read(8)
            if PMD['read'][11]:
                T[11].append(struct.unpack( \
                                CONF['endian'] + 'd', bytes)[0])

    if PMD['read'][0]:
        T[0] = np.array(T[0]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0])
    if PMD['read'][1]:
        T[1] = np.array(T[1]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0])
    if PMD['read'][2]:
        T[2] = np.array(T[2]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0])
    if PMD['read'][3]:
        T[3] = np.array(T[3]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0], \
                                        3)
    if PMD['read'][4]:
        T[4] = np.array(T[4]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0], \
                                        3)
    if PMD['read'][5]:
        T[5] = np.array(T[5]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0], \
                                        self.NRL, 2)
    if PMD['read'][6]:
        T[6] = np.array(T[6]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0], \
                                        self.NRU, 2)
    if PMD['read'][7]:
        T[7] = np.array(T[7]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0], \
                                        9)
    if PMD['read'][8]:
        T[8] = np.array(T[8]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0])
    if PMD['read'][9]:
        T[9] = np.array(T[9]).reshape(PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0])
    if PMD['read'][10]:
        T[10] = np.array(T[10]).reshape( \
                                        PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0])
    if PMD['read'][11]:
        T[11] = np.array(T[11]).reshape( \
                                        PMD['head']['nodes0'][1], \
                                        PMD['head']['nodes0'][0])

    self.pgc += 100./float(PMD['head']['nodes0'][2])
    pgc = int(self.pgc)
    if pgc > self.pgl:
        diff = pgc - self.pgl
        for ii in range(diff):
            if pgc <= 99:
                self.pg.step()
                self.pg.update_idletasks()
        self.pgl += diff

    return T


def get_data(self):
    ''' Reads the pmd for the twolevel module given the header
        Returns true if the reading went well and false otherwise.
    '''

    # Initialize progress bar
    self.pgc = 0
    self.pgl = 0

    # Check if there is a file at all
    if not os.path.isfile(PMD['head']['name']):
        return False

    # Initialize output
    inpt = {}

    # Read it as a pmd file
    try:

        self.pg.update_idletasks()

        with open(PMD['head']['name'],'rb') as f:

            f.seek(PMD['head']['size'])
            bytes = f.read(4)
            inpt['mod_ver'] = struct.unpack( \
                                CONF['endian'] + 'I', bytes)[0]
            bytes = f.read(8)
            inpt['amass'] = struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0]
            bytes = f.read(8)
            inpt['Aul'] = struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0]
            bytes = f.read(8)
            inpt['Eul'] = struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0]
            bytes = f.read(4)
            inpt['Jl2'] = struct.unpack( \
                            CONF['endian'] + 'I', bytes)[0]
            inpt['NRL'] = (inpt['Jl2'] + 1)*(inpt['Jl2'] + 1)
            self.NRL = inpt['NRL']
            bytes = f.read(4)
            inpt['Ju2'] = struct.unpack( \
                            CONF['endian'] + 'I', bytes)[0]
            inpt['NRU'] = (inpt['Ju2'] + 1)*(inpt['Ju2'] + 1)
            self.NRU = inpt['NRU']
            bytes = f.read(8)
            inpt['gl'] = struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0]
            bytes = f.read(8)
            inpt['gu'] = struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0]
            bytes = f.read(8)
            inpt['Tref'] = struct.unpack( \
                            CONF['endian'] + 'd', bytes)[0]
            inpt['nodes'] = []
            inpt['nodes'] = (PMD['head']['nodes0'][0], \
                                PMD['head']['nodes0'][1], \
                                PMD['head']['nodes0'][2])

            inpt['x'] = PMD['head']['x'] \
                                    [0:PMD['head']['nodes0'][0]]
            inpt['y'] = PMD['head']['y'] \
                                    [0:PMD['head']['nodes0'][1]]
            inpt['z'] = PMD['head']['z'] \
                                    [0:PMD['head']['nodes0'][2]]

            #
            # Compatibility with pmd files that were different
            # before 2019, but the version did not change because
            # the code was not public yet
            #

            # Read four bytes, get an integer from them, and roll
            # back the same number of bytes
            bytes = f.read(4)
            f.seek(-4,1)
            test_nx = int(struct.unpack( \
                            CONF['endian'] + 'I', bytes)[0])

            # If this integer is, in fact, nx, we jump forward
            if (test_nx == PMD['head']['nodes0'][0]):
                f.seek(8*(inpt['nodes'][0]+inpt['nodes'][1]+1),1)

            #
            # End of compatibility patch
            #

            bytes = f.read(8*inpt['nodes'][0]*
                                inpt['nodes'][1])
            inpt['tempo'] = np.array(struct.unpack( \
                            CONF['endian'] + 'd'* \
                            inpt['nodes'][0]* \
                            inpt['nodes'][1], bytes)). \
                            reshape(inpt['nodes'][1], \
                                    inpt['nodes'][0])
            inpt['nodes0'] = inpt['nodes']

            planesize = PMD['head']['nodes0'][0]* \
                        PMD['head']['nodes0'][1]* \
                        8*(22 + 2*(self.NRL + self.NRU))

            # Create geometry axis
            if CONF['interpol']:

                # Change number of nodes
                inpt['nodes'] = [int(CONF['nodes']['x']), \
                                    int(CONF['nodes']['y']), \
                                    int(CONF['nodes']['z'])]

                # Check sizes
                for ii in range(len(inpt['nodes'])):
                    if inpt['nodes'][ii] > inpt['nodes0'][ii] or \
                        inpt['nodes'][ii] <= 1:
                        raise ValueError("The dimension of the " + \
                                        "interpolated data " + \
                                        "cannot be larger than " + \
                                        "the original or " + \
                                        "smaller than 2.")

                # Original grid in Mm
                temp = {}
                temp['x'] = np.array(inpt['x'])*1e-8
                temp['y'] = np.array(inpt['y'])*1e-8
                temp['z'] = np.array(inpt['z'])*1e-8

                # Final grid
                inpt['x'] = np.linspace(min(temp['x']), \
                                        max(temp['x']), \
                                        inpt['nodes'][0])
                inpt['y'] = np.linspace(min(temp['y']), \
                                        max(temp['y']), \
                                        inpt['nodes'][1])
                inpt['z'] = np.linspace(min(temp['z']), \
                                        max(temp['z']), \
                                        inpt['nodes'][2])

                # Check what planes do we really need to read
                Flags = np.zeros(inpt['nodes0'][2], dtype=bool)
                liz1 = 0
                for iz in range(inpt['nodes0'][2]-1):

                    for iz1 in range(liz1,inpt['nodes'][2]):

                        if inpt['z'][iz1] > temp['z'][iz+1]:
                            break

                        liz1 = iz1

                        if inpt['z'][iz1] >= temp['z'][iz]:
                            Flags[iz] = Flags[iz] or True
                            Flags[iz+1] = Flags[iz+1] or True

            else:

                # Flag all planes for read
                Flags = np.ones(inpt['nodes0'][2], dtype=bool)

                # Transform into Mm
                inpt['x'] = np.array(inpt['x'])*1e-8
                inpt['y'] = np.array(inpt['y'])*1e-8
                inpt['z'] = np.array(inpt['z'])*1e-8

            # Identify scalars and vectors
            scal = [0,1,2,8,9,10,11]
            vec = [3,4]

            # Create space to read data
            for ii,read in \
                zip(range(len(PMD['read'])),PMD['read']):

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
                    elif ii == 5:
                        inpt[ii] = np.empty([inpt['nodes'][2], \
                                                inpt['nodes'][1], \
                                                inpt['nodes'][0], \
                                                self.NRL, 2])
                    elif ii == 6:
                        inpt[ii] = np.empty([inpt['nodes'][2], \
                                                inpt['nodes'][1], \
                                                inpt['nodes'][0], \
                                                self.NRU, 2])
                    elif ii == 7:
                        inpt[ii] = np.empty([inpt['nodes'][2], \
                                                inpt['nodes'][1], \
                                                inpt['nodes'][0], \
                                                9])

                else:
                    inpt[ii] = -1

            # Interpolating
            if CONF['interpol']:

                # Auxiliar initialization
                izps = 1
                NX = inpt['nodes'][0]
                NY = inpt['nodes'][1]
                P0 = {}
                P1 = {}

                for iz in range(inpt['nodes0'][2] - 1):

                    # If the next two planes are not needed,
                    # skip the current plane if have not been
                    # read
                    if not Flags[iz] or not Flags[iz+1]:
                        try:
                            check = P1['done']
                            P1 = {}
                        except KeyError:
                            P0 = self.skip_plane(f, planesize)
                        except :
                            raise
                        continue

                    # Get P0
                    try:
                        check = P0['done']
                    except KeyError:
                        P0 = self.get_plane(f)
                    except:
                        raise

                    # Get P1
                    P1 = self.get_plane(f)

                    zz0 = temp['z'][iz]
                    zz1 = temp['z'][iz+1]

                    for izp in range(izps-1,inpt['nodes'][2]):

                        izps = izp + 1
                        zz = inpt['z'][izp]

                        if zz > zz1:
                            break

                        if zz0 <= zz and zz1 >= zz:

                            # Interpolate in lower plane
                            if not P0['done']:
                                for var in scal:
                                    if not PMD['read'][var]:
                                        continue
                                    r = interpolate.interp2d( \
                                            temp['y'],temp['x'], \
                                            P0[var])
                                    P0[var] = r(inpt['x'], \
                                                inpt['y'])
                                for var in vec:
                                    if not PMD['read'][var]:
                                        continue
                                    for ii in range(3):
                                        r = interpolate. \
                                                interp2d( \
                                                temp['y'], \
                                                temp['x'], \
                                                P0[var][:,:,ii])
                                        P0[var][0:NY,0:NX,ii] = \
                                                r(inpt['x'], \
                                                    inpt['y'])
                                    P0[var] = P0[var][0:NY,0:NX,:]
                                var = 5
                                if PMD['read'][var]:
                                    for ii in range(self.NRL):
                                        for jj in range(2):
                                            r = interpolate.\
                                                    interp2d( \
                                                    temp['y'], \
                                                    temp['x'], \
                                            P0[var][:,:,ii,jj])
                                            P0[var] \
                                                [0:NY,0:NX,ii,jj] \
                                                = r(inpt['x'], \
                                                    inpt['y'])
                                    P0[var] = P0[var] \
                                                [0:NY,0:NX,:,:]
                                var = 6
                                if PMD['read'][var]:
                                    for ii in range(self.NRU):
                                        for jj in range(2):
                                            r = interpolate. \
                                                    interp2d( \
                                                    temp['y'], \
                                                    temp['x'], \
                                            P0[var][:,:,ii,jj])
                                            P0[var] \
                                                [0:NY,0:NX,ii,jj] \
                                                = r(inpt['x'], \
                                                    inpt['y'])
                                    P0[var] = P0[var] \
                                                [0:NY,0:NX,:,:]
                                var = 7
                                if PMD['read'][var]:
                                    for ii in range(9):
                                        r = interpolate. \
                                                interp2d(temp['y'], \
                                                        temp['x'], \
                                                P0[var][:,:,ii])
                                        P0[var][0:NY,0:NX,ii] = \
                                                    r(inpt['x'], \
                                                        inpt['y'])
                                    P0[var] = P0[var][0:NY,0:NX,:]
                                P0['done'] = True

                            # Interpolate in upper plane
                            if not P1['done']:
                                for var in scal:
                                    if not PMD['read'][var]:
                                        continue
                                    r = interpolate.interp2d( \
                                            temp['y'],temp['x'], \
                                            P1[var])
                                    P1[var] = r(inpt['x'], \
                                                inpt['y'])
                                for var in vec:
                                    if not PMD['read'][var]:
                                        continue
                                    for ii in range(3):
                                        r = interpolate. \
                                                interp2d( \
                                                temp['y'], \
                                                temp['x'], \
                                                P1[var][:,:,ii])
                                        P1[var][0:NY,0:NX,ii] = \
                                                r(inpt['x'], \
                                                    inpt['y'])
                                    P1[var] = P1[var][0:NY,0:NX,:]
                                var = 5
                                if PMD['read'][var]:
                                    for ii in range(self.NRL):
                                        for jj in range(2):
                                            r = interpolate.\
                                                    interp2d( \
                                                    temp['y'], \
                                                    temp['x'], \
                                            P1[var][:,:,ii,jj])
                                            P1[var] \
                                                [0:NY,0:NX,ii,jj] \
                                                = r(inpt['x'], \
                                                    inpt['y'])
                                    P1[var] = P1[var] \
                                                [0:NY,0:NX,:,:]
                                var = 6
                                if PMD['read'][var]:
                                    for ii in range(self.NRU):
                                        for jj in range(2):
                                            r = interpolate. \
                                                    interp2d( \
                                                    temp['y'], \
                                                    temp['x'], \
                                            P1[var][:,:,ii,jj])
                                            P1[var] \
                                                [0:NY,0:NX,ii,jj] \
                                                = r(inpt['x'], \
                                                    inpt['y'])
                                    P1[var] = P1[var] \
                                                [0:NY,0:NX,:,:]
                                var = 7
                                if PMD['read'][var]:
                                    for ii in range(9):
                                        r = interpolate. \
                                                interp2d(temp['y'], \
                                                        temp['x'], \
                                                P1[var][:,:,ii])
                                        P1[var][0:NY,0:NX,ii] = \
                                                    r(inpt['x'], \
                                                        inpt['y'])
                                    P1[var] = P1[var][0:NY,0:NX,:]
                                P1['done'] = True


                            # Interpolate in vertical
                            w1 = (zz1 - zz)/(zz1 - zz0)
                            w2 = 1 - w1
                            for var in scal:
                                if not PMD['read'][var]:
                                    continue
                                inpt[var][izp,:,:] = \
                                            w1*P0[var] + w2*P1[var]
                            for var in vec:
                                if not PMD['read'][var]:
                                    continue
                                for ii in range(3):
                                    inpt[var][izp,:,:,ii] = \
                                            w1*P0[var][:,:,ii] + \
                                            w2*P1[var][:,:,ii]
                            var = 5
                            if PMD['read'][var]:
                                for ii in range(self.NRL):
                                    for jj in range(2):
                                        inpt[var] \
                                            [izp,:,:,ii,jj] = \
                                            w1*P0[var][:,:,ii,jj] + \
                                            w2*P1[var][:,:,ii,jj]
                            var = 6
                            if PMD['read'][var]:
                                for ii in range(self.NRU):
                                    for jj in range(2):
                                        inpt[var] \
                                            [izp,:,:,ii,jj] = \
                                            w1*P0[var][:,:,ii,jj] + \
                                            w2*P1[var][:,:,ii,jj]
                            var = 7
                            if PMD['read'][var]:
                                for ii in range(9):
                                    inpt[var][izp,:,:,ii] = \
                                        w1*P0[var][:,:,ii] + \
                                        w2*P1[var][:,:,ii]

                    if iz < inpt['nodes0'][2] - 2:

                        # Shift planes
                        if Flags[iz+2]:
                            P0 = copy.deepcopy(P1)
                        else:
                            P0 = {}
                    
                    else:

                        P1 = {}
                        P0 = {}

            # Do not interpolate
            else:

                for iz in range(inpt['nodes'][2]):

                    P0 = self.get_plane(f)

                    for ii in scal:
                        if PMD['read'][ii]:
                            inpt[ii][iz,:,:] = P0[ii]
                    for ii in (vec+[5,6,7]):
                        if PMD['read'][ii]:
                            inpt[ii][iz,:,:,:] = P0[ii]

            f.close()

            PMD['data'] = inpt
            return True

    except:

        raise
        return False

if __name__ == '__main__':
    inpt = {}
    filein = '../../DATA/graphnet_nlte/AR_385_CAII.pmd'
    with open(filein,'rb') as f:
        inpt['io'] = "binary"

        inpt['size'] = 0
        bytes = f.read(8)
        inpt['size'] += 8
        if sys.version_info[0] < 3:
            inpt['magic'] = ''.join(struct.unpack( CONF['endian'] + 'c'*8, bytes))
        else:
            inpt['magic'] = ''
            for ii in range(8):
                byte = bytes[ii:ii+1].decode('utf-8')
                inpt['magic'] += byte
        if inpt['magic'] != 'portapmd':
            msg = "Magic string {0} not recognized"
            msg = msg.format(inpt['magic'])
            raise ValueError(msg)

        bytes = f.read(1)
        inpt['size'] += 1
        endian = struct.unpack('<b', bytes)[0]
        if endian == 1:
            inpt['endian'] = '>'
        elif endian == 0:
            inpt['endian'] = '<'
        else:
            msg = "Endian {0} not recognized"
            msg = msg.format(endian)
            raise ValueError(msg)

        bytes = f.read(1)
        inpt['size'] += 1
        inpt['isize'] = struct.unpack(inpt['endian'] + 'b', bytes)[0]

        bytes = f.read(1)
        inpt['size'] += 1
        inpt['dsize'] = struct.unpack(inpt['endian'] + 'b', bytes)[0]

        bytes = f.read(4)
        inpt['size'] += 4
        inpt['pmd_ver'] = struct.unpack(inpt['endian'] + 'I', bytes)[0]

        bytes = f.read(24)
        inpt['size'] += 24
        inpt['date'] = struct.unpack(inpt['endian'] + 'I'*6, bytes)

        bytes = f.read(2)
        inpt['size'] += 2
        inpt['period'] = struct.unpack(inpt['endian'] + 'bb', bytes)

        bytes = f.read(24)
        inpt['size'] += 24
        inpt['domain'] = struct.unpack(inpt['endian'] + 'ddd', bytes)

        bytes = f.read(24)
        inpt['size'] += 24
        inpt['origin'] = struct.unpack(inpt['endian'] + 'ddd', bytes)

        bytes = f.read(12)
        inpt['size'] += 12
        inpt['nodes'] = struct.unpack(inpt['endian'] + 'III', bytes)
        inpt['nodes0'] = inpt['nodes']

        bytes = f.read(8192*8)
        inpt['size'] += 8192*8
        inpt['x'] = struct.unpack(inpt['endian'] + 'd'*8192, bytes)[0:inpt['nodes'][0]]

        bytes = f.read(8192*8)
        inpt['size'] += 8192*8
        inpt['y'] = struct.unpack(inpt['endian'] + 'd'*8192, bytes)[0:inpt['nodes'][1]]

        bytes = f.read(8192*8)
        inpt['size'] += 8192*8
        inpt['z'] = struct.unpack(inpt['endian'] + 'd'*8192, bytes)[0:inpt['nodes'][2]]

        inpt['angles'] = []
        bytes = f.read(4)
        inpt['size'] += 4
        inpt['angles'].append(struct.unpack(inpt['endian'] + 'I', bytes)[0])

        bytes = f.read(4)
        inpt['size'] += 4
        inpt['angles'].append(struct.unpack(inpt['endian'] + 'I', bytes)[0])

        bytes = f.read(1023)
        inpt['size'] += 1023
        if sys.version_info[0] < 3:
            module = struct.unpack(inpt['endian'] + \
                                    'c'*1023, bytes)
            inpt['module'] = ['']
            for i in range(1023):
                if module[i] == '\x00' or module[i] == '\n':
                    break
                else:
                    inpt['module'].append(module[i])
            inpt['module'] = ''.join(inpt['module'])
        else:
            inpt['module'] = ''
            for i in range(1023):
                byte = bytes[i:i+1].decode('utf-8')
                if byte == '\x00' or byte == '\n':
                    break
                else:
                    inpt['module'] += byte

        bytes = f.read(4096)
        inpt['size'] += 4096
        if sys.version_info[0] < 3:
            comments = struct.unpack(inpt['endian'] + \
                                        'c'*4096, bytes)
            inpt['comments'] = ['']
            for i in range(4096):
                if comments[i] == '\x00':
                    break
                else:
                    inpt['comments'].append(comments[i])
            inpt['comments'] = ''.join(inpt['comments'])
        else:
            inpt['comments'] = ''
            for i in range(4096):
                byte = bytes[i:i+1].decode('utf-8')
                if byte == '\x00':
                    break
                else:
                    inpt['comments'] += byte
        # Strip ends of line
        inpt['comments'] = inpt['comments'].strip()

        # Check if everything in comment is a space
        if inpt['comments'] == \
            ' '*len(list(inpt['comments'])):
            inpt['comments'] = ' '

        # Introduce changes of lines
        else:
            # Spaces to add
            ns = 20
            # First, split in jumps
            Tmp = inpt['comments'].split('\n')
            # For each section
            for ii in range(len(Tmp)):
                # Add 10 spaces at each section after the
                # first
                if ii > 0:
                    Tmp[ii] = ' '*ns + Tmp[ii]
                siz = len(list(Tmp[ii]))
                col = 400
                # If big size of comment
                while col < siz:
                    Tmp[ii] = Tmp[ii][:col+1] + '\n' + \
                                ' '*ns + Tmp[ii][col+1:]
                    col += 402 + ns
                    siz += 2 + ns
            # Mount back the comments
            inpt['comments'] = '\n'.join(Tmp)

        bytes = f.read(4)
        inpt['size'] += 4
        inpt['msize'] = struct.unpack(inpt['endian'] + 'I', bytes)[0]

        bytes = f.read(4)
        inpt['size'] += 4
        inpt['gsize'] = struct.unpack(inpt['endian'] + 'I', bytes)[0]

        inpt['loaded'] = False
        path, filename = os.path.split(filein)
        inpt['name'] = filein
        inpt['nameshort'] = "{0}".format(filename)

        # Check file size knowing module
        check = check_size(inpt)
        if not check:
            raise ValueError("File size is not correct")

        inpt['pmd_dir'] = path

