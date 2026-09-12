import pickle
import os

files = ['Iwave', 'logdeparture', 'n_Nat', 'ne', 'T', 'tau', 'vlos', 'vturb', 'z']
folders = ['data_1d_si_v2', 'data_1d_si_v2_2']
destination_folder = 'data_1d_si_v2_joined'
prefixes = ['train', 'test', 'validation']

if not os.path.exists(f'../data/{destination_folder}'):
    os.makedirs(f'../data/{destination_folder}')


for prefix in prefixes:
    for file in files:
        dataset = []
        for folder in folders:
            with open(f'../data/{folder}/{prefix}_{file}.pkl', 'rb') as filehandle:
                print(f'reading file: \t ../data/{folder}/{prefix}_{file}.pkl')
                loaded = pickle.load(filehandle)
                dataset += loaded.copy()
                del loaded

            with open(f'../data/{destination_folder}/{prefix}_{file}.pkl', 'wb') as filehandle:
                pickle.dump(dataset, filehandle)
