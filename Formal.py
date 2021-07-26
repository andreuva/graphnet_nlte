from configobj import ConfigObj
import numpy as np
import time
from tqdm import tqdm
import glob
import os
import pickle

import torch
import torch_geometric.data
import torch.nn as nn
import shutil

import graphnet
from Dataset import Dataset as dtst


try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


# Class that will containg the GN as well as methods for training, testing and predicting
class Formal(object):
    def __init__(self, configuration='conf.dat', batch_size=64, gpu=0,
                 smooth=0.05, validation_split=0.2, datadir='', predict=False):

        # Is a GPU available?
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        # Factor to be used for smoothing the loss with an exponential window
        self.smooth = smooth

        # If the nvidia_smi package is installed, then report some additional information
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(
                self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))

        self.batch_size = batch_size
        self.kwargs = {'num_workers': 1, 'pin_memory': False} if self.cuda else {}

        if not predict:
            # Read the configuration file
            f = open(configuration, 'r')
            tmp = f.readlines()
            f.close()

            # Parse configuration file and transform to integers
            self.hyperparameters = ConfigObj(tmp)

            for k, q in self.hyperparameters.items():
                self.hyperparameters[k] = int(q)

            self.datadir = datadir
            self.validation_split = validation_split

            # Instantiate the model with the hyperparameters
            self.model = graphnet.EncodeProcessDecode(**self.hyperparameters).to(self.device)

    def optimize(self, savedir, epochs, lr=3e-4):

        # Print the number of trainable parameters
        print('N. total trainable parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Instantiate the dataset
        self.dataset = dtst(self.hyperparameters, self.datadir)

        # Randomly shuffle a vector with the indices to separate between training/validation datasets
        idx = np.arange(self.dataset.n_training)
        np.random.shuffle(idx)

        self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)

        # Define the data loaders
        self.train_loader = torch_geometric.data.DataLoader(
            self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **self.kwargs)
        self.validation_loader = torch_geometric.data.DataLoader(
            self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **self.kwargs)

        self.lr = lr
        self.n_epochs = epochs

        # Define the name of the model
        filename = time.strftime("%Y%m%d-%H%M%S")
        print(' Model: {0}'.format(savedir + filename))

        # Copy model
        shutil.copyfile(graphnet.__file__,
                        '{0}.model.py'.format(savedir + filename))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Cosine annealing learning rate scheduler. This will reduce the learning rate with a cosing law
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Now start the training
        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):

            filename = time.strftime("%Y%m%d-%H%M%S")
            # Compute training and validation steps
            train_loss = self.train(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            # If the validation loss improves, save the model as best
            if (valid_loss < best_loss):
                best_loss = valid_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters': self.hyperparameters,
                    'optimizer': self.optimizer.state_dict(),
                }

                print("Saving best model...")
                torch.save(checkpoint, savedir + filename + '_best.pth')

            # Update the learning rate
            self.scheduler.step()

    def train(self, epoch):

        # Put the model in training mode
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0

        for batch_idx, (data) in enumerate(t):

            # Extract the node, edges, indices, target, global and batch information from the Data class
            node = data.x
            edge_attr = data.edge_attr
            edge_index = data.edge_index
            target = data.y
            u = data.u
            batch = data.batch

            # Move them to the GPU
            node, edge_attr, edge_index = node.to(self.device), edge_attr.to(
                self.device), edge_index.to(self.device)
            u, batch, target = u.to(self.device), batch.to(
                self.device), target.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Evaluate Graphnet
            out = self.model(node, edge_attr, edge_index, u, batch)

            # Compute loss
            loss = self.loss_fn(out.squeeze(), target.squeeze())

            # Compute backpropagation
            loss.backward()

            # Update the parameters
            self.optimizer.step()

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            # Compute smoothed loss
            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            # Update information for this batch
            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu,
                              memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)

        return loss_avg

    def validate(self):
        # Do a validation of the model and return the loss

        self.model.eval()
        loss_avg = 0
        t = tqdm(self.validation_loader)
        with torch.no_grad():
            for batch_idx, (data) in enumerate(t):

                node = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                target = data.y
                u = data.u
                batch = data.batch

                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(
                    self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(
                    self.device), target.to(self.device)

                out = self.model(node, edge_attr, edge_index, u, batch)

                loss = self.loss_fn(out.squeeze(), target.squeeze())

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

                t.set_postfix(loss=loss_avg)

        return loss_avg

    def test(self, checkpoint=None, readir='../weights/', savedir='../test/', dtst_type='validation'):
        # test the model with a given dataset and save the results

        if (checkpoint is None):
            files = glob.glob(readir + '*.pth')
            self.checkpoint = sorted(files)[-1]
        else:
            self.checkpoint = '{0}.pth'.format(checkpoint)

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)

        self.hyperameters = checkpoint['hyperparameters']
        self.test_model = graphnet.EncodeProcessDecode(**self.hyperameters).to(self.device)
        self.test_model.load_state_dict(checkpoint['state_dict'])
        self.test_dataset = dtst(self.hyperameters, self.datadir, dtst_type)

        self.test_loader = torch_geometric.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)

        # Loss function
        self.test_loss_fn = nn.MSELoss()

        self.test_model.eval()
        tq = tqdm(self.test_loader)

        self.test_target = []
        self.test_out = []
        self.test_T = []
        self.u = []
        loss_avg = np.array([])

        with torch.no_grad():
            for data in tq:

                node = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                target = data.y
                u = data.u
                batch = data.batch

                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(
                    self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(
                    self.device), target.to(self.device)

                out = self.test_model(node, edge_attr, edge_index, u, batch)

                loss_avg = np.append(loss_avg, self.test_loss_fn(out.squeeze(), target.squeeze()).item())

                n = len(data.ptr) - 1
                for i in range(n):
                    left = data.ptr[i]
                    right = data.ptr[i+1]
                    self.test_out.append(out[left:right, :].cpu().numpy())
                    self.test_target.append(target[left:right, :].cpu().numpy())
                    self.test_T.append(node[left:right, :].cpu().numpy())
                    self.u.append(u[left:right, :].cpu().numpy())

        print(f'Average test loss: {loss_avg.mean()}\n')

        print(f'SAVING THE PREDICTIONS, TARGETS AND FEATURES IN: {savedir}test_%time%.pkl')

        test_dict = {'target': self.test_target,
                     'prediction': self.test_out,
                     'features': self.test_T,
                     'global': self.u,
                     'loss': loss_avg,
                     'checkpoint': self.checkpoint,
                     'datadir': self.datadir,
                     'hyperparams': self.hyperameters
                     }

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        with open(savedir + f'{dtst_type}_checkpoint_{self.checkpoint[-24:-9]}_at_{time.strftime("%Y%m%d-%H%M%S")}.pkl', 'wb') as filehandle:
            pickle.dump(test_dict, filehandle)

    def predict(self, TT=[None], tau=[None], vturb=[None], vlos=[None], ne=[None], cmass=[None], checkpoint=None, readir=None):
        # Prediction of simple models directly and return the results

        if (checkpoint is None):
            if readir is None:
                raise ValueError('Not checkpoint or read directory selected')
            files = glob.glob(readir + '*.pth')
            self.checkpoint = sorted(files)[-1]
        else:
            self.checkpoint = checkpoint

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)

        self.hyperameters = checkpoint['hyperparameters']
        self.predict_model = graphnet.EncodeProcessDecode(**self.hyperameters).to(self.device)
        self.predict_model.load_state_dict(checkpoint['state_dict'])

        print("=> constructing the dataset to predict")
        directory = 'tmp/'
        prefix = 'tmp'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # print('The created temporary directory is %s' % directory)
        # print('And inside it the created temporary datasets:')
        with open(directory + f'{prefix}_vturb.pkl', 'wb') as filehandle:
            pickle.dump(vturb, filehandle)

        with open(directory + f'{prefix}_T.pkl', 'wb') as filehandle:
            pickle.dump(TT, filehandle)

        with open(directory + f'{prefix}_tau.pkl', 'wb') as filehandle:
            pickle.dump(tau, filehandle)

        with open(directory + f'{prefix}_cmass.pkl', 'wb') as filehandle:
            pickle.dump(cmass, filehandle)

        with open(directory + f'{prefix}_vlos.pkl', 'wb') as filehandle:
            pickle.dump(vlos, filehandle)

        with open(directory + f'{prefix}_ne.pkl', 'wb') as filehandle:
            pickle.dump(ne, filehandle)

        logdep = []
        for temperature in TT:
            logdep.append(np.zeros((len(temperature), self.hyperameters['output_size'])))

        with open(directory + f'{prefix}_logdeparture.pkl', 'wb') as filehandle:
            # print(directory + f'{prefix}_logdeparture.pkl')
            pickle.dump(logdep, filehandle)

        print("=> Loading the dataset to predict")
        self.pred_dataset = dtst(self.hyperameters, directory, prefix)

        # Remove the temporary files and directory
        [os.remove(file) for file in glob.glob(directory + '*')]
        os.rmdir(directory)

        self.pred_loader = torch_geometric.data.DataLoader(
            self.pred_dataset, batch_size=1, shuffle=False, **self.kwargs)

        # Loss function
        self.pred_loss_fn = nn.MSELoss()

        self.predict_model.eval()
        print("=> Making the predictions")
        tq = tqdm(self.pred_loader)

        self.pred_out = []

        with torch.no_grad():
            for data in tq:

                node = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                target = data.y
                u = data.u
                batch = data.batch

                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(self.device), target.to(self.device)

                out = self.predict_model(node, edge_attr, edge_index, u, batch)

                n = len(data.ptr) - 1
                for i in range(n):
                    left = data.ptr[i]
                    right = data.ptr[i+1]
                    self.pred_out.append(out[left:right, :].cpu().numpy()*5)

        return self.pred_out
