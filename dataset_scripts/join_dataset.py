import pickle
import os

files = ['logdeparture', 'cmass', 'tau', 'T', 'vturb']
folders = ['prd', 'prd_2']
destination_folder = 'prd_full'

if not os.path.exists(f'../data/{destination_folder}'):
    os.makedirs(f'../data/{destination_folder}')

for file in files:
    dataset = []
    for folder in folders:
        with open(f'../data/{folder}/train_{file}.pkl', 'rb') as filehandle:
            print(f'reading file: \t ../data/{folder}/train_{file}.pkl')
            loaded = pickle.load(filehandle)
            dataset += loaded.copy()
            del loaded

        with open(f'../data/{destination_folder}/train_{file}.pkl', 'wb') as filehandle:
            pickle.dump(dataset, filehandle)
