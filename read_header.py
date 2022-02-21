import struct
import os,sys
from data_playground import VARS

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