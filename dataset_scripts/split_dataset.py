import pickle
import os

files = ['logdeparture', 'cmass', 'Iwave', 'ne', 'n_Nat', 'tau', 'T', 'vlos', 'vturb']
folder = 'crd_vlos'
destination_folder = 'crd_vlos_split'

if not os.path.exists(f'../data/{destination_folder}'):
    os.makedirs(f'../data/{destination_folder}')

for file in files:
    with open(f'../data/{folder}/train_{file}.pkl', 'rb') as filehandle:
        print(f'reading file: \t ../data/{folder}/train_{file}.pkl')
        dtst = pickle.load(filehandle)

    with open(f'../data/{destination_folder}/train_{file}.pkl', 'wb') as filehandle:
        pickle.dump(dtst[:int(len(dtst)/2)], filehandle)

    with open(f'../data/{destination_folder}/train_{file}_2.pkl', 'wb') as filehandle:
        pickle.dump(dtst[int(len(dtst)/2):], filehandle)
