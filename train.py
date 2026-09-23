import os
import shutil
import time
import argparse
from Formal import Formal


try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int, metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float, metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int, metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=64, type=int, metavar='BATCH', help='Batch size')
    parser.add_argument('--conf', '--conf', default='conf.dat', type=str, metavar='CONF', help='Configuration file')
    parser.add_argument('--rd', '--readir', default=f'../data/train/', metavar='READIR', help='directory for reading the training data (train_* and validation_* pickles)')
    parser.add_argument('--sav', '--savedir', default=f'../checkpoints/', metavar='SAVEDIR', help='directory for output files')
    parser.add_argument('--seed', default=0, type=int, metavar='SEED', help='seed for parameter init and batch order')
    parser.add_argument('--compile', action='store_true', help='torch.compile the network (~5 min compile, ~1.5x faster steps)')
    parser.add_argument('--resume', default=None, metavar='RUN_DIR', help='existing run directory to resume from its last.pth')

    parsed = vars(parser.parse_args())

    if parsed['resume']:
        # Continue an interrupted run in place: same directory, same conf.dat, same code copies
        run_dir = parsed['resume']
        configuration = os.path.join(run_dir, 'conf.dat')
        resume = os.path.join(run_dir, 'last.pth')
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(parsed['sav'], timestamp)
        configuration = parsed['conf']
        resume = None

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        files_to_copy = [
            os.path.join(script_dir, 'Dataset.py'),
            os.path.join(script_dir, 'Formal.py'),
            os.path.join(script_dir, 'graphnet.py'),
            os.path.join(script_dir, 'train.py'),
        ]

        for file_path in files_to_copy:
            if os.path.exists(file_path):
                shutil.copy(file_path, run_dir)

        # The configuration actually used, always stored as <run_dir>/conf.dat so that
        # --resume reads the right one whatever the original file was called.
        shutil.copy(parsed['conf'], os.path.join(run_dir, 'conf.dat'))

    savedir_run = os.path.join(run_dir, '')

    network = Formal(
                     configuration=configuration,
                     batch_size=parsed['batch'],
                     gpu=parsed['gpu'],
                     smooth=parsed['smooth'],
                     datadir=parsed['rd'],
                     seed=parsed['seed'],
                     compile=parsed['compile'])

    network.optimize(savedir_run, parsed['epochs'], lr=parsed['lr'], resume=resume)

