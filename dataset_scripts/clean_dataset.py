import numpy as np
import pickle
from glob import glob

dir = '../data/prd/test_*.pkl'
files = glob(dir)
print('selected the files:')
[print(file) for file in files]

old_dtst = [None] * len(files)
new_dtst = [None] * len(files)

while True:
    for i, file in enumerate(files):
        with open(file, 'rb') as filehandle:
            old_dtst[i] = pickle.load(filehandle)

    indx_nans = []
    for dtst in old_dtst:
        for j, depart in enumerate(dtst):
            if np.isnan(np.sum(depart)):
                indx_nans.append(j)

    for i, dtst in enumerate(old_dtst):
        clean = dtst.copy()
        for indx in indx_nans:
            del clean[indx]
        new_dtst[i] = clean

    if len(indx_nans) == 0:
        break

    for i, file in enumerate(files):
        with open(file, 'wb') as filehandle:
            pickle.dump(new_dtst[i], filehandle)
