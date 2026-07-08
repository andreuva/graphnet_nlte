import numpy as np

datadir = '/dat/andreuva/gpu/graphnet/data_train'
pops = np.memmap(f'{datadir}/AR_385_CaII_5L_pops.dat', dtype='<f4', mode='r', shape=(425, 504, 504, 6))
beta = np.load(f'{datadir}/AR_385_CaII_5L_beta.npy')

print("pops shape:", pops.shape)
print("beta shape:", beta.shape)

n_lte = pops / (beta + 1e-12)

print("pops range:", np.min(pops), np.max(pops))
print("beta range:", np.min(beta), np.max(beta))
print("n_lte range:", np.min(n_lte), np.max(n_lte))
