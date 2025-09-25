import numpy as np
from random import shuffle
from astropy.convolution import Box1DKernel
from astropy.convolution import convolve
import scipy.io as io
import scipy.interpolate as interp
from tqdm import tqdm
from glob import glob
import os
import pickle
import time
import sys
from mpi4py import MPI
import argparse
from enum import IntEnum
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, \
    Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw


class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3


def smooth(sig, kernel=Box1DKernel, width=2):
    " Function to smooth out a signal with a kernel "
    return convolve(sig, kernel(width))


def synth_spectrum(atmos, depthData=False, Nthreads=1, conserveCharge=False, prd=False):

    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)

    # Configure the set of atomic models to use.
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom(), Al_atom(), CaII_atom(),
                            Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])

    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    # aSet.set_active('H', 'Ca')
    aSet.set_active('H', 'Ca', 'Si')

    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # compute the equilibrium populations at the fixed electron density provided in the model
    eqPops = aSet.compute_eq_pops(atmos)

    # Configure the Context which holds the state of the simulation for the
    # backend, and provides the python interface to the backend.
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=Nthreads, conserveCharge=conserveCharge)

    if depthData:
        ctx.depthData.fill = True

    # Iterate the Context to convergence
    lw.iterate_ctx_se(ctx, prd=prd, quiet=True)

    # Update the background populations based on the converged solution and
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    # compute the final solution for mu=1 on the provided wavelength grid.
    ctx.formal_sol_gamma_matrices()
    if prd:
        ctx.prd_redistribute()
    return ctx

""" 
def iterate_ctx_crd(ctx, prd=False, Nscatter=10, NmaxIter=500):
    '''
    Iterate a Context to convergence.
    '''
    for i in range(NmaxIter):
        # Compute the formal solution
        dJ = ctx.formal_sol_gamma_matrices()
        if prd:
            ctx.prd_redistribute()
        # Just update J for Nscatter iterations
        if i < Nscatter:
            continue
        # Update the active populations under statistical equilibrium,
        # conserving charge if this option was set on the Context.
        delta = ctx.stat_equil()

        # If we are converged in both relative change of J and populations return
        if dJ < 3e-3 and delta < 1e-3:
            return
 """

class Model_generator(object):

    def __init__(self, train, datadir):
        """Loading of all the data in models of atmospheres (BIFORST + ATMOSREF)"""

        self.train = train
        # Read all the needed atmospheric models to compute the samples
        print(f"READING BIFROST: ...\n", flush=True)
        if self.train < 0:
            self.bifrost = io.readsav(datadir + 'snap530_rh.save')
        else:
            self.bifrost = io.readsav(datadir + 'snap385_rh.save')

        # store the number of models in the bifrost dataset and the current state
        self.n_bifrost = self.bifrost['tg'][0, :, :].size

        self.bifrost['tg'] = np.reshape(self.bifrost['tg'], (self.bifrost['tg'].shape[0], -1))
        self.bifrost['vlos'] = np.reshape(self.bifrost['vlos'], (self.bifrost['vlos'].shape[0], -1))
        self.bifrost['nel'] = np.reshape(self.bifrost['nel'], (self.bifrost['nel'].shape[0], -1))

        # Shufle the bifrost columns
        index = np.arange(0, self.bifrost['tg'].shape[-1])
        shuffle(index)
        self.bifrost['tg'] = self.bifrost['tg'][:, index]
        self.bifrost['vlos'] = self.bifrost['vlos'][:, index]
        self.bifrost['nel'] = self.bifrost['nel'][:, index]

        # selecting different parts of the simulation if we are in training or testing
        if self.train < 0:
            self.current_bifrost = 0
        elif self.train > 0:
            self.current_bifrost = 0
            self.n_bifrost = int(self.n_bifrost*0.8)
        else:
            self.current_bifrost = int(0.8*self.n_bifrost)

        # If we are in training or testing read the semiempirical models if not we just use bifrost
        if self.train >= 0:
            print(f"READING ATMOSPHERES: ...\n", flush=True)
            # Read all the models in the datadir folder and store it in the atmosRef list
            atmospheres = sorted(glob(datadir + '*.atmos'))
            self.atmosRef = [None] * len(atmospheres)

            for i, atmos in enumerate(atmospheres):
                print(f"Reading {os.path.split(atmos)[-1]}", flush=True)
                _, atmos_i = lw.multi.read_multi_atmos(atmos)
                self.atmosRef[i] = atmos_i

            # define it's length
            self.n_ref_atmos = len(self.atmosRef)

            # Print completion message and flush to ensure it's seen immediately
            print(f"Finished reading {self.n_ref_atmos} atmospheric models\n", flush=True)

            # Define the lists to store the arrays of taus, Temperatures, and other atmosphere variables
            # Log_10(tau) in the atmospheres
            self.ltau = [None] * self.n_ref_atmos
            # Log_10(tau) in the nodes
            self.ltau_nodes = [None] * self.n_ref_atmos
            # Number of taus in the nodes
            self.ntau = [None] * self.n_ref_atmos
            # indexes at wich ltau_nodes insert sorted in ltau
            self.ind_ltau = [None] * self.n_ref_atmos
            # temperatures in the atmospheres
            self.logT = [None] * self.n_ref_atmos

            # Define the arrays for each reference atmosphere
            for i in range(self.n_ref_atmos):
                self.ltau[i] = np.log10(self.atmosRef[i].tauRef)
                self.ltau_nodes[i] = np.array([np.min(self.ltau[i]), -5, -4, -3, -2, -1, 0, np.max(self.ltau[i])])
                self.ntau[i] = len(self.ltau_nodes[i])
                self.ind_ltau[i] = np.searchsorted(self.ltau[i], self.ltau_nodes[i]) - 1
                self.logT[i] = np.log10(self.atmosRef[i].temperature)
            
            print(f"Finished Model generator initialization\n", flush=True)

    def new_model(self):
        """Method to read the parameters of an atmosphere based on 1 random refence atmosphere and perturbing it
        to obtain diferent results or from BIFROST snapshot"""

        """ Pick randomly a sample from bifrost or from the reference atmospheres unless we already
        computed all the bifrost models """
        if self.train < 0:
            choices = [True]
        else:
            choices = [True, False]

        if np.random.choice(a=choices) and self.current_bifrost < self.n_bifrost:

            # Read the model parameters
            heigth = np.float64(self.bifrost['z'][::-1]*1e3)
            T_new = np.float64(self.bifrost['tg'][:, self.current_bifrost][::-1])
            vlos_new = np.float64(self.bifrost['vlos'][:, self.current_bifrost][::-1])
            vturb_new = vlos_new*0
            ne = np.float64(self.bifrost['nel'][:, self.current_bifrost][::-1])

            # Set the depth as the tau500 and the depth scale acordingly
            depth = heigth
            depth_scale = lw.ScaleType.Geometric

            # increase the number of processed bifrost models
            self.current_bifrost += 1

            return depth_scale, depth, T_new, vlos_new, vturb_new, ne

        else:
            # pick one reference atmosphere
            i = np.random.randint(low=0, high=self.n_ref_atmos)

            # Define the std and compute the normal distribution to perturb the ref.atmosphere
            std = 2500
            deltas = np.random.normal(loc=0.0, scale=std, size=self.ntau[i])

            # smooth the deltas convolving with a box function of width 2 (func smooth at the begining of file)
            deltas_smooth = smooth(deltas)
            # interpolate the Temperature at the values of ltau and add a delta contribution
            f = interp.interp1d(self.ltau_nodes[i], deltas_smooth, kind='quadratic', bounds_error=False, fill_value="extrapolate")
            T_new = self.atmosRef[i].temperature + f(self.ltau[i])
            # if T_new < 2500K set it to 2500K
            T_new[T_new < 2500] = 2500

            # Perturb vturb by 20% of the current value
            std = 0.2*self.atmosRef[i].vturb[self.ind_ltau[i]]
            deltas_vturb = np.random.normal(loc=0.0, scale=std, size=self.ntau[i])
            f = interp.interp1d(self.ltau_nodes[i], deltas_vturb, kind='quadratic', bounds_error=False, fill_value="extrapolate")
            vturb_new = self.atmosRef[i].vturb + f(self.ltau[i])
            ne = None

            # Set the v LOS to 0 + perturbations
            std = 2500
            deltas_vlos = np.random.normal(loc=0.0, scale=std, size=self.ntau[i])
            f = interp.interp1d(self.ltau_nodes[i], deltas_vlos, kind='quadratic', bounds_error=False, fill_value="extrapolate")
            vlos_new = 0 + f(self.ltau[i])

            # Select the depth as the column mass and the depth scale acordingly
            depth = self.atmosRef[i].cmass
            depth_scale = lw.ScaleType.ColumnMass

            return depth_scale, depth, T_new, vlos_new, vturb_new, ne


def master_work(nsamples, train, prd_active, savedir, readdir, filename, write_frequency=1):
    """ Function to define the work to do by the master """
    # Calling the Model_generator to read the models and initialice the class
    mg = Model_generator(train, readdir)

    # Index of the task to keep track of each job
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    # Define the lists that will store the data of each feature-label pair
    log_departure_list = [None] * nsamples      # Departure coeficients b = log(n/n*) LABEL
    n_Nat_list = [None] * nsamples              # population of the lebel/total population
    T_list = [None] * nsamples                  # Temperatures
    tau_list = [None] * nsamples                # optical depths
    vturb_list = [None] * nsamples              # Turbulent velocities
    vlos_list = [None] * nsamples              # line of sight velocities
    cmass_list = [None] * nsamples              # Column mass
    ne_list = [None] * nsamples                 # density of electrons in the atmosphere
    Iwave_list = [None] * nsamples              # Intensity profile of the model

    success = True

    tasks_status = [0] * nsamples

    print("Master starting loop to distribute the work", flush=True)

    # loop to compute the nsamples pairs
    with tqdm(total=nsamples, ncols=110, disable=(rank != 0), file=sys.stdout) as pbar:
        # While we don't have more closed workers than total workers keep looping
        while closed_workers < num_workers:
            # Recieve the data from any process that says it's alive and get wich one is and it's status
            dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            # print(" * MASTER: received data from worker {0} with tag {1}.".format(source, tag), flush=True)

            # if the worker is ready to work send them a task
            if tag == tags.READY:
                # print(" * MASTER: worker {0} is ready.".format(source), flush=True)
                # Worker is ready, so send it a task
                try:
                    # select the first index with status is 0
                    task_index = tasks_status.index(0)

                    # Ask the model generator for a new model and send the data to the process to compute the atmos. and NLTE pop.
                    depth_scale, depth, T, vlos, vturb, ne = mg.new_model()

                    dataToSend = {'index': task_index, 'prd_active': prd_active, 'ne': ne,
                                  'depth_scale': depth_scale, 'depth': depth, 'T': T, 'vlos': vlos, 'vturb': vturb}

                    # send the data of the task and put the status tu 1 (done)
                    # print(f" * MASTER: sending task {task_index} to worker {source}.", flush=True)
                    comm.send(dataToSend, dest=source, tag=tags.START)
                    tasks_status[task_index] = 1

                # If this not work set the tag of the worker to exit and kill it
                except ValueError as e:
                    # print(f"!!! MASTER: no more tasks to distribute, telling worker {source} to exit. !!!", flush=True)
                    comm.send(None, dest=source, tag=tags.EXIT)
                except Exception as e:
                    import traceback
                    print(f"!!! MASTER ERROR sending task to worker {source} !!!")
                    print(f"Exception Type: {type(e).__name__}, Message: {e}")
                    traceback.print_exc()
                    print("!!! Telling worker to exit. !!!")
                    comm.send(None, dest=source, tag=tags.EXIT)

            # If the tag it's Done, recieve the status, the index and all the data
            # and update the progress bar
            elif tag == tags.DONE:
                # print(" * MASTER: worker {0} has completed task {1}.".format(source, dataReceived['index']), flush=True)
                index = dataReceived['index']
                success = dataReceived['success']

                if (not success):
                    # print(f"!!! MASTER: worker {source} failed to compute task {index}, rescheduling. !!!", flush=True)
                    tasks_status[index] = 0

                else:
                    log_departure_list[index] = dataReceived['log_departure']
                    n_Nat_list[index] = dataReceived['n_Nat']
                    T_list[index] = dataReceived['T']
                    tau_list[index] = dataReceived['tau']
                    vturb_list[index] = dataReceived['vturb']
                    vlos_list[index] = dataReceived['vlos']
                    cmass_list[index] = dataReceived['cmass']
                    ne_list[index] = dataReceived['ne']
                    Iwave_list[index] = dataReceived['Iwave']
                    pbar.update(1)
                    # pbar.refresh()
                    # sys.stdout.flush()
                    print(f" * MASTER: task {index} completed from worker {source} ({pbar.n}/{nsamples})", flush=True)

            # if the worker has the exit tag mark it as closed.
            elif tag == tags.EXIT:
                # print(" * MASTER : worker {0} exited.".format(source))
                closed_workers += 1

            # If the number of itterations is multiple with the write frequency dump the data
            if (pbar.n / write_frequency == pbar.n // write_frequency):

                with open(savedir + f'{filename}_logdeparture.pkl', 'wb') as filehandle:
                    pickle.dump(log_departure_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_n_Nat.pkl', 'wb') as filehandle:
                    pickle.dump(n_Nat_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_T.pkl', 'wb') as filehandle:
                    pickle.dump(T_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_vturb.pkl', 'wb') as filehandle:
                    pickle.dump(vturb_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_vlos.pkl', 'wb') as filehandle:
                    pickle.dump(vlos_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_tau.pkl', 'wb') as filehandle:
                    pickle.dump(tau_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_cmass.pkl', 'wb') as filehandle:
                    pickle.dump(cmass_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_ne.pkl', 'wb') as filehandle:
                    pickle.dump(ne_list[0:task_index], filehandle)

                with open(savedir + f'{filename}_Iwave.pkl', 'wb') as filehandle:
                    pickle.dump(Iwave_list[0:task_index], filehandle)

    # Once finished, dump all the data
    print("Master finishing")

    with open(savedir + f'{filename}_cmass.pkl', 'wb') as filehandle:
        pickle.dump(cmass_list, filehandle)

    with open(savedir + f'{filename}_logdeparture.pkl', 'wb') as filehandle:
        pickle.dump(log_departure_list, filehandle)

    with open(savedir + f'{filename}_n_Nat.pkl', 'wb') as filehandle:
        pickle.dump(n_Nat_list, filehandle)

    with open(savedir + f'{filename}_T.pkl', 'wb') as filehandle:
        pickle.dump(T_list, filehandle)

    with open(savedir + f'{filename}_vturb.pkl', 'wb') as filehandle:
        pickle.dump(vturb_list, filehandle)

    with open(savedir + f'{filename}_vlos.pkl', 'wb') as filehandle:
        pickle.dump(vlos_list, filehandle)

    with open(savedir + f'{filename}_tau.pkl', 'wb') as filehandle:
        pickle.dump(tau_list, filehandle)

    with open(savedir + f'{filename}_ne.pkl', 'wb') as filehandle:
        pickle.dump(ne_list, filehandle)

    with open(savedir + f'{filename}_Iwave.pkl', 'wb') as filehandle:
        pickle.dump(Iwave_list, filehandle)


def slave_work(rank):
    # Function to define the work that the slaves will do

    while True:
        # Send the master the signal that the worker is ready
        comm.send(None, dest=0, tag=tags.READY)
        # recieve the data with the index of the task, the atmosphere parameters and/or the tag
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        # print(" * WORKER {0}: received data from master with tag {1}.".format(rank, tag), flush=True)

        if tag == tags.START:
            # Recieve the model atmosphere to solve the NLTE problem
            task_index = dataReceived['index']
            prd_active = dataReceived['prd_active']
            depth = dataReceived['depth']
            depth_scale = dataReceived['depth_scale']
            temperature = dataReceived['T']
            vlos = dataReceived['vlos']
            vturb = dataReceived['vturb']
            ne = dataReceived['ne']

            # Initialice the variables in case the convergence fails send None
            log_departure = None
            n_Nat = None
            tau = None
            cmass = None
            wave = np.linspace(853.9444, 854.9444, 1001)
            Iwave = wave*0
            success = 1
            # print(f" * WORKER {rank}: starting task {task_index}.", flush=True)

            try:
                # Compute the new atmosphere and solve the NLTE problem and retrieve the solved parameters
                atmos = lw.Atmosphere.make_1d(scale=depth_scale, depthScale=depth, temperature=temperature,
                                              vlos=vlos, vturb=vturb, verbose=False, ne=ne)
                ctx = synth_spectrum(atmos, depthData=True, conserveCharge=False, prd=prd_active)
                # print(f" * WORKER {rank}: finished NLTE task {task_index}.", flush=True)
                tau = atmos.tauRef
                cmass = atmos.cmass
                temperature = atmos.temperature
                ne = atmos.ne
                vturb = atmos.vturb
                vlos = atmos.vlos
                # Compute the departure coefficients
                log_departure = np.log10(ctx.activeAtoms[0].n / ctx.activeAtoms[0].nStar)
                n_Nat = np.log10(ctx.activeAtoms[0].n / ctx.activeAtoms[0].nTotal)
                n_active = len(ctx.activeAtoms)
                if n_active > 1:
                    for at in range(1, n_active):
                        log_departure = np.append(log_departure,
                                                  np.log10(ctx.activeAtoms[at].n / ctx.activeAtoms[at].nStar),
                                                  axis=0)
                        n_Nat = np.append(n_Nat,
                                          np.log10(ctx.activeAtoms[at].n / ctx.activeAtoms[at].nTotal),
                                          axis=0)

                Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)

                # If the coefficients are not converged set as failure
                if np.isnan(np.sum(log_departure)):
                    # print(f"!!! WORKER {rank}: task {task_index} did not converge !!!", flush=True)
                    success = 0

            except Exception as e:
                import traceback
                print(f"!!! WORKER {rank}: Exception in task {task_index} !!!", flush=True)
                print(f"Exception Type: {type(e).__name__}, Message: {e}")
                traceback.print_exc()
                print(f"!!! WORKER {rank}: task {task_index} failed !!!", flush=True)
                success = 0

            # Send the computed data
            dataToSend = {'index': task_index, 'T': temperature, 'log_departure': log_departure, 'n_Nat': n_Nat,
                          'tau': tau, 'cmass': cmass, 'vlos': vlos, 'vturb': vturb, 'ne': ne, 'success': success, 'Iwave': Iwave}
            comm.send(dataToSend, dest=0, tag=tags.DONE)
            # print(f" * WORKER {rank}: finished task {task_index}.", flush=True)

        # If the tag is exit break the loop and kill the worker and send the EXIT tag to master
        elif tag == tags.EXIT:
            # print(f" * WORKER {rank}: exiting.", flush=True)
            break

    comm.send(None, dest=0, tag=tags.EXIT)


if (__name__ == '__main__'):

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    print(f"Node {rank}/{size} active", flush=True)

    if rank == 0:
        parser = argparse.ArgumentParser(description='Generate synthetic models and solve NLTE problem')
        parser.add_argument('--n', '--nmodels', default=10000, type=int, metavar='NMODELS', help='Number of models')
        parser.add_argument('--train', '--tr', default=0, type=int, metavar='TRAINING', help='Flag for computing training or test datasets')
        parser.add_argument('--f', '--freq', default=10, type=int, metavar='FREQ', help='Frequency of model write')
        parser.add_argument('--sav', '--savedir', default=f'../data/{time.strftime("%Y%m%d-%H%M%S")}/', metavar='SAVEDIR', help='directory for output files')
        parser.add_argument('--rd', '--readir', default=f'../data/models_atmos/', metavar='READIR', help='directory for reading files')
        parser.add_argument('--prd', '--prd', default=0, type=int, metavar='PRD', help='partial redistribution flag')
        # parser.add_argument('--o', '--out', default='train', metavar='OUTFILE', help='Root of output files')

        parsed = vars(parser.parse_args())

        if not os.path.exists(parsed['sav']):
            os.makedirs(parsed['sav'])

        if parsed['train'] < 0:
            print('Computing VALIDATION dataset')
            filename = 'validation'
        elif parsed['train'] > 0:
            print('Computing TRAINING dataset')
            filename = 'train'
        else:
            print('Computing TESTING dataset')
            filename = 'test'

        if parsed['prd']:
            print('with PRD')

        # Wait for all processes to be ready before starting work
        print("Master: Waiting for all workers to be ready...", flush=True)
        comm.barrier()
        print("Master: Starting work distribution...", flush=True)

        master_work(parsed['n'], parsed['train'], parsed['prd'], parsed['sav'], parsed['rd'], filename, write_frequency=parsed['f'])
    else:
        # Worker processes wait for master to finish initialization
        print(f"Worker {rank}: Waiting for master to finish initialization...", flush=True)
        comm.barrier()
        print(f"Worker {rank}: Starting slave work...", flush=True)
        slave_work(rank)
