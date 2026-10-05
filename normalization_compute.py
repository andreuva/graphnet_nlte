# run from graphnet_nlte/ in the gph environment
import pickle
import numpy as np

def load(key):
    with open(f'../data_1d_si_v4/train_{key}.pkl', 'rb') as f:
        return pickle.load(f)

z = load('z')
print(f'{len(z)} columns, {np.mean([len(c) == 211 for c in z]):.1%} with 211 points')
feats = {'T_log10': np.log10(np.concatenate(load('T'))),
         'z': np.concatenate(z),
         'tau_log10': np.log10(np.concatenate(load('tau'))),
         'ne_log10': np.log10(np.concatenate(load('ne'))),
         'vturb_km': np.concatenate(load('vturb')) / 1e3,
         'vlos_km': np.concatenate(load('vlos')) / 1e3}
for name, x in feats.items():
    print(f"    '{name}': {{'mean': {x.mean():.6g}, 'std': {x.std():.6g}}},")
dz = np.concatenate([np.diff(c) for c in z])
print(f"    'delta_z': {{'std': {np.sqrt(np.mean(dz ** 2)):.6g}}},")
