from configobj import ConfigObj
import numpy as np
import time
from tqdm import tqdm
import glob
import os
import pickle
import random

import torch
import torch_geometric.data
import torch.nn as nn
import shutil

import graphnet
from Dataset import Dataset as dtst
from Dataset import NORM_STATS


# Maximum gradient L2 norm allowed through to the optimizer, and Adam's numerical parameters.
# Per-parameter gradients here are ~3e-6, small enough that the default eps=1e-8 with
# beta2=0.999 leaves the second-moment estimate stale between updates -- the regime in which one
# unlucky batch produces an oversized step. A larger eps and a shorter beta2 memory are the
# standard mitigation; both are conservative and cost nothing.
GRAD_CLIP_NORM = 0.5
ADAM_EPS = 1e-6
ADAM_BETAS = (0.9, 0.95)

# The learning rate ramps linearly from this fraction of --lr to --lr over the first epoch
# (stepped per batch), then follows a cosine decay to zero over the remaining epochs.
WARMUP_START_FACTOR = 0.01


def masked_mse(out, target, mask=None):
    """
    Mean squared error over the unmasked entries only.

    `mask` is a 1/0 float tensor the same shape as `target`, zero on the (depth, level) points
    whose population is astrophysically negligible -- see Dataset.NEGLIGIBLE_LOG_N_OVER_NTOT.
    Averaging over the surviving entries rather than over all of them keeps the loss comparable
    between batches with different amounts of masking.
    """
    sq = (out - target) ** 2
    if mask is None:
        return sq.mean()
    mask = mask.to(sq.dtype).view_as(sq)
    return (sq * mask).sum() / mask.sum().clamp(min=1.0)


try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


# Class that will containg the GN as well as methods for training, testing and predicting
class Formal(object):
    def __init__(self, configuration='conf.dat', batch_size=64, gpu=0,
                 smooth=0.05, datadir='', predict=False, seed=0, compile=False):

        # Seed everything that draws random numbers in training: parameter init, the shuffle
        # order of the training loader, and any numpy/python randomness.
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Is a GPU available?
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        if self.cuda:
            # TF32 tensor cores for fp32 matmuls (~15% on an H100 for this model).
            torch.set_float32_matmul_precision('high')
        self.compile = compile

        # Factor to be used for smoothing the loss with an exponential window
        self.smooth = smooth

        # If the nvidia_smi package is installed, then report some additional information
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(
                self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))

        self.batch_size = batch_size
        self.kwargs = {'num_workers': 12, 'pin_memory': True} if self.cuda else {}

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

            # Instantiate the model with the hyperparameters
            self.model = graphnet.EncodeProcessDecode(**self.hyperparameters).to(self.device)
            # `forward_model` is what train/validate call; `model` keeps the plain module for
            # state_dict. torch.compile(dynamic=True) takes ~5 min to compile this network
            # and then runs a training step ~1.5x faster (kernel-launch bound otherwise).
            self.forward_model = torch.compile(self.model, dynamic=True) if self.compile else self.model

    def optimize(self, savedir, epochs, lr=3e-4, resume=None):

        # Print the number of trainable parameters
        print('N. total trainable parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Instantiate the datasets. Training uses the whole train_* split and validation the
        # on-disk validation_* split (a different Bifrost snapshot). Carving validation out of
        # train_* at random put near-copies of spatially correlated training columns into it,
        # and the loss that selected the best checkpoint read ~2x lower than on the true hold-out.
        self.dataset = dtst(self.hyperparameters, self.datadir, 'train')
        self.valid_dataset = dtst(self.hyperparameters, self.datadir, 'validation')

        # Define the data loaders
        self.train_loader = torch_geometric.loader.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed), **self.kwargs)
        self.validation_loader = torch_geometric.loader.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)

        self.lr = lr
        self.n_epochs = epochs

        # Define the name of the model
        filename = time.strftime("%Y%m%d-%H%M%S")
        print(' Model: {0}'.format(savedir + filename))

        # Copy model
        shutil.copyfile(graphnet.__file__,
                        '{0}.model.py'.format(savedir + filename))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          betas=ADAM_BETAS, eps=ADAM_EPS)

        # Learning rate schedule, stepped once per batch: linear warmup over the first epoch,
        # then cosine annealing to zero over the remaining steps.
        n_steps = self.n_epochs * len(self.train_loader)
        warmup_steps = max(1, min(len(self.train_loader), n_steps // 2))
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            [torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=WARMUP_START_FACTOR,
                                               total_iters=warmup_steps),
             torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, max(1, n_steps - warmup_steps))],
            milestones=[warmup_steps])

        # Loss function: plain MSE restricted to the levels/depths that carry population
        self.loss_fn = masked_mse

        # Now start the training
        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')
        first_epoch = 1

        if resume is not None:
            print(f"=> resuming from '{resume}'")
            last = torch.load(resume, map_location=self.device, weights_only=False)
            self.model.load_state_dict(last['state_dict'])
            self.optimizer.load_state_dict(last['optimizer'])
            self.scheduler.load_state_dict(last['scheduler'])
            self.train_loss = last['train_loss']
            self.valid_loss = last['valid_loss']
            best_loss = last['best_loss']
            first_epoch = last['epoch'] + 1

        for epoch in range(first_epoch, epochs + 1):

            # Compute training and validation steps
            train_loss = self.train(epoch)
            valid_loss = self.validate()

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'train_loss': self.train_loss,
                'valid_loss': self.valid_loss,
                'best_loss': min(best_loss, valid_loss),
                'hyperparameters': self.hyperparameters,
                # Normalization constants used by Dataset.py to build this checkpoint's
                # training inputs/targets, so inference code can always reproduce the
                # exact normalization a given checkpoint was trained with, even after
                # NORM_STATS is later retuned for a new training run.
                'norm_stats': NORM_STATS,
            }

            # If the validation loss improves, overwrite best.pth (weights only, no optimizer)
            if (valid_loss < best_loss):
                best_loss = valid_loss
                print("Saving best model...")
                torch.save(checkpoint, savedir + 'best.pth')

            # last.pth is the resume point: weights plus optimizer and scheduler state,
            # overwritten every epoch (see train.py --resume).
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['scheduler'] = self.scheduler.state_dict()
            torch.save(checkpoint, savedir + 'last.pth')

    def train(self, epoch):

        # Put the model in training mode
        self.model.train()
        print("\nEpoch {0}/{1}".format(epoch, self.n_epochs))
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
            mask = getattr(data, 'mask', None)

            # Move them to the GPU
            node, edge_attr, edge_index = node.to(self.device), edge_attr.to(
                self.device), edge_index.to(self.device)
            u, batch, target = u.to(self.device), batch.to(
                self.device), target.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Evaluate Graphnet
            out = self.forward_model(node, edge_attr, edge_index, u, batch)

            # Compute loss
            loss = self.loss_fn(out.squeeze(), target.squeeze(), mask)

            # Compute backpropagation
            loss.backward()

            # Clip before stepping. The processor is ~600 layers deep (n_message_passing_steps
            # blocks of two 3-layer MLPs), which makes it prone to the occasional oversized Adam
            # step: the loss jumps 10-40x for part of an epoch and then recovers, which is what
            # the spikes in the epoch-loss curve are. Measured gradient norms on a converged
            # checkpoint are ~0.02 (p99 0.05), so this threshold never binds in normal operation
            # and only truncates the rare event.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)

            # Update the parameters and the learning rate (per-batch schedule)
            self.optimizer.step()
            self.scheduler.step()

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
        # Do a validation of the model and return the loss.
        #
        # This is a true mean over the whole held-out set, weighted by how many entries each
        # batch contributes, not the exponentially-smoothed running value used during training.
        # The smoothed version had a memory of ~20 batches out of ~770 and the loader draws them
        # in a fresh random order every epoch, so the number that decides which checkpoint gets
        # saved as "best" was an average over a random ~2.6% of the validation set.

        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0
        t = tqdm(self.validation_loader)
        with torch.no_grad():
            for batch_idx, (data) in enumerate(t):

                node = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                target = data.y
                u = data.u
                batch = data.batch
                mask = getattr(data, 'mask', None)

                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(
                    self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(
                    self.device), target.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                out = self.forward_model(node, edge_attr, edge_index, u, batch)

                loss = self.loss_fn(out.squeeze(), target.squeeze(), mask)

                # Each batch's loss is already a mean over its own unmasked entries, so weight by
                # that count to recover the mean over the whole set.
                weight = target.numel() if mask is None else mask.sum().item()
                total_loss += loss.item() * weight
                total_weight += weight

                t.set_postfix(loss=total_loss / max(total_weight, 1.0))

        return total_loss / max(total_weight, 1.0)

    def test(self, checkpoint=None, readir='../weights/', savedir='../test/', dtst_type='validation'):
        # test the model with a given dataset and save the results

        if (checkpoint is None):
            files = glob.glob(readir + '*best.pth')
            self.checkpoint = sorted(files)[-1]
        else:
            self.checkpoint = '{0}.pth'.format(checkpoint)

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage, weights_only=False)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage, weights_only=False)

        self.hyperameters = checkpoint['hyperparameters']
        self.test_model = graphnet.EncodeProcessDecode(**self.hyperameters).to(self.device)
        self.test_model.load_state_dict(checkpoint['state_dict'])
        # Use the constants this checkpoint was trained with, not whatever NORM_STATS currently
        # holds, so an older checkpoint is still fed the inputs it expects.
        self.test_dataset = dtst(self.hyperameters, self.datadir, dtst_type,
                                 norm_stats=checkpoint.get('norm_stats'))

        self.test_loader = torch_geometric.loader.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)

        # Loss function: same masked MSE the model was trained with
        self.test_loss_fn = masked_mse

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
                mask = getattr(data, 'mask', None)

                node, edge_attr, edge_index = node.to(self.device), edge_attr.to(
                    self.device), edge_index.to(self.device)
                u, batch, target = u.to(self.device), batch.to(
                    self.device), target.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                out = self.test_model(node, edge_attr, edge_index, u, batch)

                loss_avg = np.append(loss_avg, self.test_loss_fn(out.squeeze(), target.squeeze(), mask).item())

                n = len(data.ptr) - 1
                for i in range(n):
                    left = data.ptr[i]
                    right = data.ptr[i+1]
                    self.test_out.append(out[left:right, :].cpu().numpy())
                    self.test_target.append(target[left:right, :].cpu().numpy())
                    self.test_T.append(node[left:right, :].cpu().numpy())
                    # u carries one row per graph; data.ptr indexes nodes, so slicing u with it
                    # returned the wrong rows (harmless only while u is identically zero).
                    self.u.append(u[i, :].cpu().numpy())

        print(f'Average test loss: {loss_avg.mean()}\n')

        print(f'SAVING THE PREDICTIONS, TARGETS AND FEATURES IN: {savedir}test_%time%.pkl')

        test_dict = {'target': self.test_target,
                     'prediction': self.test_out,
                     'features': self.test_T,
                     'global': self.u,
                     'loss': loss_avg,
                     'checkpoint': self.checkpoint,
                     'train_loss': checkpoint.get('train_loss', None),
                     'valid_loss': checkpoint.get('valid_loss', None),
                     'datadir': self.datadir,
                     'hyperparams': self.hyperameters
                     }

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        run_name = os.path.basename(os.path.dirname(os.path.abspath(self.checkpoint)))
        with open(savedir + f'{dtst_type}_checkpoint_{run_name}_at_{time.strftime("%Y%m%d-%H%M%S")}.pkl', 'wb') as filehandle:
            pickle.dump(test_dict, filehandle)

    def predict(self, TT=[None], tau=[None], vturb=[None], vlos=[None], ne=[None], zz=[None], checkpoint=None, readir=None):
        # Prediction of simple models directly and return the results

        if (checkpoint is None):
            if readir is None:
                raise ValueError('Not checkpoint or read directory selected')
            files = glob.glob(readir + '*best.pth')
            self.checkpoint = sorted(files)[-1]
        else:
            self.checkpoint = checkpoint

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage, weights_only=False)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage, weights_only=False)

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

        with open(directory + f'{prefix}_z.pkl', 'wb') as filehandle:
            pickle.dump(zz, filehandle)

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
        self.pred_dataset = dtst(self.hyperameters, directory, prefix,
                                 norm_stats=checkpoint.get('norm_stats'))

        # Remove the temporary files and directory
        [os.remove(file) for file in glob.glob(directory + '*')]
        os.rmdir(directory)

        self.pred_loader = torch_geometric.loader.DataLoader(
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
