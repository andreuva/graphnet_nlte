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
