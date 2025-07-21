import numpy as np
import matplotlib.pyplot as plt
import os, mmap, pathlib

# ---- grid dimensions taken from the C code ----
nx = ny = 504
nz = 476 - 52 + 1          # 425
nlev = 6                   # caii[0] … caii[5]

# ---- memory–mapped array: reads only the chunks you touch ----
pops = np.memmap(
    '../../data_porta/AR_385_CaII_5L_pops.dat',
    dtype='<f4',           # little-endian 32-bit float
    mode='r',
    shape=(nz, ny, nx, nlev)   # (k, j, i, level)
)

# example: population of level 0 at plane k=0 (z-index 52), full horizontal map
pop_L0_plane0 = pops[0, :, :, 0]        # shape (504, 504)

# example: vertical profile through the centre (i=j=252)
central_column = pops[:, 252, 252, :5]  # shape (425, 5)  – only the 5 useful levels
total_population = np.sum(central_column, axis=1)

# plot the central column populations
plt.figure(figsize=(10, 6))
plt.plot(central_column[:, 0]/total_population, label='L0', marker='o')
plt.plot(central_column[:, 1]/total_population, label='L1', marker='o')
plt.plot(central_column[:, 2]/total_population, label='L2', marker='o')
plt.plot(central_column[:, 3]/total_population, label='L3', marker='o')
plt.plot(central_column[:, 4]/total_population, label='L4', marker='o')
plt.title('Vertical Profile of Ca II Populations at Centre (i=j=252)')
plt.xlabel('Vertical Index (k)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()
