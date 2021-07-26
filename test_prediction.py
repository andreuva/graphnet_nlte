import argparse
import os
from Formal import Formal


try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Test neural network')

    parser.add_argument('--dtst', '--dataset', default='validation', type=str, metavar='DTST', help='Type of dataset to predict')
    parser.add_argument('--gpu', '--gpu', default=0, type=int, metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float, metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--batch', '--batch', default=64, type=int, metavar='BATCH', help='Batch size')
    parser.add_argument('--split', '--split', default=0.2, type=float, metavar='SPLIT', help='Validation split')
    parser.add_argument('--conf', '--conf', default='conf.dat', type=str, metavar='CONF', help='Configuration file')
    parser.add_argument('--rd', '--readir', default=f'../data/database/', metavar='READIR', help='directory for reading the training data')
    parser.add_argument('--sav', '--savedir', default=f'../checkpoints/savedir/', metavar='SAVEDIR', help='directory for output files')
    parser.add_argument('--testdir', '--testdir', default=f'../test/testGN/', metavar='TESTDIR', help='directory for test files')

    parsed = vars(parser.parse_args())

    if not os.path.exists(parsed['sav']):
        os.makedirs(parsed['sav'])

    network = Formal(
                     configuration=parsed['conf'],
                     batch_size=parsed['batch'],
                     gpu=parsed['gpu'],
                     validation_split=parsed['split'],
                     smooth=parsed['smooth'],
                     datadir=parsed['rd'])

    network.test(readir=parsed['sav'], savedir=parsed['testdir'], dtst_type=parsed['dtst'])
