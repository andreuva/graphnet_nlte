# departure_coefficients_calculation.py

import numpy as np
import json
import sys
from configobj import ConfigObj

# Append firtez-nlte path to import physical classes
sys.path.append('../../firtez-nlte')
from atoms import MultiLevelAtom
from lte_utils import compute_lte_populations

# 1. Load config
config = ConfigObj('conf.dat', unrepr=True)
datadir = config['data']['datadir']
nx = config['data']['nx']
ny = config['data']['ny']
nz = config['data']['nz_orig']
nlev = config['data']['nlev']

print("--- Loaded Configuration ---")
print(f"Data directory: {datadir}")
print(f"Grid size: {nz}x{ny}x{nx}")
print(f"Number of levels: {nlev}")
print("----------------------------\n")

# 2. Load original memory-mapped arrays
print("Loading memory-mapped data from the original grid...")
pops = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, nlev))
temp = np.memmap(f'{datadir}/AR_385_temp.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
n_e = np.memmap(f'{datadir}/AR_385_ne.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))
n_h = np.memmap(f'{datadir}/AR_385_nh.dat', dtype='<f4', mode='r', shape=(nz, ny, nx, 1))

print(f"Populations shape: {pops.shape}")
print(f"Temperature shape: {temp.shape}")
print(f"Electron density shape: {n_e.shape}")
print(f"Hydrogen density shape: {n_h.shape}\n")

# Squeeze to remove singleton dimensions for compute_lte_populations
temp_squeezed = np.squeeze(temp)
ne_squeezed = np.squeeze(n_e)
nh_squeezed = np.squeeze(n_h)

# 3. Load Ca II Atom configuration
atom_config_path = '../../firtez-nlte/config_H_Ca_Mg_Na.json'
print(f"Loading Ca II atom configuration from {atom_config_path}...")
with open(atom_config_path) as f:
    full_config = json.load(f)
    
ca_config = None
for atom_dict in full_config['atoms']:
    if atom_dict['name'] == 'Ca_II':
        ca_config = atom_dict
        break
        
if ca_config is None:
    raise ValueError("Ca_II atom configuration not found in config_H_Ca_Mg_Na.json")
    
atom = MultiLevelAtom.from_dict(ca_config)
print("Successfully instantiated MultiLevelAtom for Ca II.\n")

# 4. Compute LTE populations
print("Computing LTE populations using vectorized compute_lte_populations...")
# Convert ne and nh from SI (m^-3) to CGS (cm^-3) for compute_lte_populations
# Also, calculate total hydrogen density instead of using neutral hydrogen nh
nh_total_squeezed = np.where(temp_squeezed > 20000.0, nh_squeezed + ne_squeezed / 1.2, nh_squeezed + ne_squeezed)
ne_cgs = ne_squeezed * 1e-6
nh_cgs = nh_total_squeezed * 1e-6

pops_lte_cgs = compute_lte_populations(atom, temp_squeezed, ne_cgs, nh_cgs)
print(f"LTE populations (CGS) shape: {pops_lte_cgs.shape}\n")

# 5. Compute Departure Coefficients (beta)
epsilon = 1e-32
print("Computing departure coefficients (beta = pops_nlte_cgs / (pops_lte_cgs + epsilon))...")
# Since the raw pops file is in CGS (cm^-3), we divide it directly by pops_lte_cgs
beta = pops / (pops_lte_cgs + epsilon)

# 6. Save beta to file
output_path = f'{datadir}/AR_385_CaII_5L_beta.npy'
print(f"Saving departure coefficients to {output_path}...")
np.save(output_path, beta.astype(np.float32))
print("Calculation completed successfully!\n")

