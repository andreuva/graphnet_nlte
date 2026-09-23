import numpy as np
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
    Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
# Si_atom_custom is not in released lightweaver -- it ships with this repo (si_atom.py).
from si_atom import Si_atom_custom
import lightweaver as lw


class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3


# Fraction of the snapshot's x-extent reserved for the training split; the rest is the test
# split. The hold-out is a contiguous strip in x, and is derived from the column's position in
# the cube rather than from a random draw, for two reasons:
#   - it is deterministic, so a `--train 1` run and a `--train 0` run of this script always
#     produce a true partition. The previous scheme reshuffled the columns with an unseeded
#     random.shuffle in every run and then sliced [0:0.8N] / [0.8N:N] of *different*
#     permutations, which put ~80% of the test columns back into the training set;
#   - neighbouring columns of a granulation snapshot are strongly correlated (a granule spans
#     ~15 pixels of a 504x504 cube), so even a correctly partitioned random column split leaves
#     near-copies of training columns in the test set. A strip breaks that correlation at a
#     single boundary.
# Validation uses a different snapshot entirely (snap530) and so is allowed the whole cube.
TRAIN_X_FRACTION = 0.8

# Iteration cap handed to lw.iterate_ctx_se. It returns normally when it runs out of iterations
# rather than raising, so synth_spectrum has to compare against this to notice.
NMAX_ITER = 2000


def smooth(sig, kernel=Box1DKernel, width=2):
    " Function to smooth out a signal with a kernel "
    return convolve(sig, kernel(width))


def synth_spectrum(atmos, depthData=False, Nthreads=1, conserveCharge=False, prd=False):

    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)

    # Configure the set of atomic models to use.
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), OI_ord_atom(), Si_atom_custom(), Al_atom(), CaII_atom(),
                            Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])

    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    # aSet.set_active('H', 'Ca')
    aSet.set_active('Si')

    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # compute the equilibrium populations at the fixed electron density provided in the model
    eqPops = aSet.compute_eq_pops(atmos)

    # Configure the Context which holds the state of the simulation for the
    # backend, and provides the python interface to the backend.
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=Nthreads, conserveCharge=conserveCharge)

    if depthData:
        ctx.depthData.fill = True

    # Iterate the Context to convergence. iterate_ctx_se returns normally after NmaxIter whether
    # or not it converged, and quiet=True suppresses the message that says so, so a non-converged
    # solve would otherwise be written into the database as long as the populations stayed finite.
    # Raising here puts it on slave_work's existing failure path, which reschedules the sample.
    niter = lw.iterate_ctx_se(ctx, prd=prd, quiet=True, NmaxIter=NMAX_ITER)
    if niter >= NMAX_ITER - 1:
        raise RuntimeError(f'statistical equilibrium did not converge in {NMAX_ITER} iterations')

    # Update the background populations based on the converged solution and
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    # compute the final solution on the provided wavelength grid. Note that the emergent
    # intensity is later evaluated at atmos.muz[-1] = 0.9531 (theta = 17.6 deg), the outermost
    # node of the 5-point quadrature -- not at disk centre.
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

    def __init__(self, train, datadir, seed=1234):
        """Loading of all the data in models of atmospheres (BIFORST + ATMOSREF)"""

        self.train = train
        # Read all the needed atmospheric models to compute the samples
        print(f"READING BIFROST: ...\n", flush=True)
        if self.train < 0:
            self.bifrost = io.readsav(datadir + 'snap530_rh.save')
        else:
            self.bifrost = io.readsav(datadir + 'snap385_rh.save')

        # Cube layout before flattening: (nz, nx, ny). We need nx/ny to map a flattened column
        # index back to its (ix, iy) position in the snapshot, which is what defines the split.
        _, nx, ny = self.bifrost['tg'].shape

        self.bifrost['tg'] = np.reshape(self.bifrost['tg'], (self.bifrost['tg'].shape[0], -1))
        self.bifrost['vlos'] = np.reshape(self.bifrost['vlos'], (self.bifrost['vlos'].shape[0], -1))
        self.bifrost['nel'] = np.reshape(self.bifrost['nel'], (self.bifrost['nel'].shape[0], -1))

        # Select which columns of the snapshot this split is allowed to draw from. Flattening is
        # C-ordered, so the column at flat index i sits at ix = i // ny (see TRAIN_X_FRACTION).
        columns = np.arange(nx * ny)
        if self.train < 0:
            allowed = columns                       # validation: different snapshot, use it all
        else:
            in_train_strip = (columns // ny) < int(TRAIN_X_FRACTION * nx)
            allowed = columns[in_train_strip] if self.train > 0 else columns[~in_train_strip]

        # Only the *order* in which the allowed columns are consumed is randomised, and with a
        # fixed seed, so that a run that is stopped early is still a reproducible and spatially
        # representative sample of its own strip. Indexing through this array instead of
        # permuting the cubes also avoids three full copies of the snapshot. The same generator
        # drives every other draw in new_model() (branch choice, reference-atmosphere choice and
        # perturbations), so a database is fully determined by (--train, --n, --seed) up to the
        # order in which workers return; previously those used the unseeded global numpy RNG.
        self.rng = np.random.default_rng(seed)
        self.column_order = self.rng.permutation(allowed)
        self.n_bifrost = self.column_order.size
        self.current_bifrost = 0

        split_name = {-1: 'validation', 0: 'test', 1: 'train'}[int(np.sign(self.train))]
        print(f"BIFROST split '{split_name}': {self.n_bifrost} of {nx*ny} columns "
              f"(cube {nx}x{ny}, seed {seed})\n", flush=True)

        # Read the semiempirical reference atmospheres for every split, including validation, so
        # that validation samples also get vturb/vlos perturbations instead of being Bifrost-only
        # (Bifrost columns always carry vturb=0, see new_model() below).
        if True:
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
                # Knots spread evenly in log(tau) between the two ends of *this* model. The
                # previous fixed ladder [min, -5, -4, -3, -2, -1, 0, max] assumed the models reach
                # tau500 = 1; none of them does (they top out between log tau = -2.3 and -2.9), so
                # the array was not monotonic. interp1d silently re-sorted x and y together, three
                # of the eight knots fell outside the model and did no work, and the interpolant
                # near the lower boundary was driven by a knot beyond it. An even spread reproduces
                # the intended ~1 dex knot spacing while being strictly increasing by construction.
                self.ltau_nodes[i] = np.linspace(self.ltau[i].min(), self.ltau[i].max(), 8)
                self.ntau[i] = len(self.ltau_nodes[i])
                # clip: searchsorted returns 0 for the first knot, so the -1 used to wrap round to
                # the *last* depth point, giving the top knot the vturb of the bottom of the model.
                self.ind_ltau[i] = np.clip(
                    np.searchsorted(self.ltau[i], self.ltau_nodes[i]) - 1, 0, len(self.ltau[i]) - 1)
                self.logT[i] = np.log10(self.atmosRef[i].temperature)
            
            print(f"Finished Model generator initialization\n", flush=True)

    def _perturbed_vturb(self, i):
        """vturb of reference atmosphere i, perturbed by 20% at the knots (on its own depth grid)."""
        std = 0.2*self.atmosRef[i].vturb[self.ind_ltau[i]]
        deltas_vturb = self.rng.normal(loc=0.0, scale=std, size=self.ntau[i])
        f = interp.interp1d(self.ltau_nodes[i], deltas_vturb, kind='quadratic', bounds_error=False, fill_value="extrapolate")
        vturb_new = self.atmosRef[i].vturb + f(self.ltau[i])
        # The quadratic interpolation overshoots and can drive vturb negative. Only vturb**2
        # ever reaches the physics, so a negative value is silently folded to its magnitude by
        # lightweaver -- but it is fed to the network *signed*, making two identical
        # atmospheres look like different inputs, and api.compute_dep_coeffs rejects it.
        vturb_new[vturb_new < 0] = 0.0
        return vturb_new

    def new_model(self):
        """Method to read the parameters of an atmosphere based on 1 random refence atmosphere and perturbing it
        to obtain diferent results or from BIFROST snapshot"""

        """ Pick randomly a sample from bifrost or from the reference atmospheres unless we already
        computed all the bifrost models """
        choices = [True, False]

        if self.rng.choice(choices) and self.current_bifrost < self.n_bifrost:

            # Read the model parameters
            column = self.column_order[self.current_bifrost]
            heigth = np.float64(self.bifrost['z'][::-1]*1e3)
            T_new = np.float64(self.bifrost['tg'][:, column][::-1])
            vlos_new = np.float64(self.bifrost['vlos'][:, column][::-1])
            ne = np.float64(self.bifrost['nel'][:, column][::-1])

            # Bifrost resolves its velocity field and carries no microturbulence, while every
            # reference atmosphere does, so vturb = 0 made this single feature a perfect label for
            # the data branch. Give the column the perturbed vturb stratification of a random
            # reference atmosphere instead, interpolated in height (held constant beyond the
            # reference model's own z range), so both branches draw vturb from the same family.
            i = self.rng.integers(self.n_ref_atmos)
            z_ref = self.atmosRef[i].z
            vturb_new = np.interp(heigth, z_ref[::-1], self._perturbed_vturb(i)[::-1])

            # Set the depth as the tau500 and the depth scale acordingly
            depth = heigth
            depth_scale = lw.ScaleType.Geometric

            # increase the number of processed bifrost models
            self.current_bifrost += 1

            return depth_scale, depth, T_new, vlos_new, vturb_new, ne

        else:
            # pick one reference atmosphere
            i = self.rng.integers(self.n_ref_atmos)

            # Define the std and compute the normal distribution to perturb the ref.atmosphere
            std = 2500
            deltas = self.rng.normal(loc=0.0, scale=std, size=self.ntau[i])

            # smooth the deltas convolving with a box function of width 2 (func smooth at the begining of file)
            deltas_smooth = smooth(deltas)
            # interpolate the Temperature at the values of ltau and add a delta contribution
            f = interp.interp1d(self.ltau_nodes[i], deltas_smooth, kind='quadratic', bounds_error=False, fill_value="extrapolate")
            T_new = self.atmosRef[i].temperature + f(self.ltau[i])
            # if T_new < 2500K set it to 2500K
            T_new[T_new < 2500] = 2500

            # Perturb vturb by 20% of the current value
            vturb_new = self._perturbed_vturb(i)
            ne = None

            # Set the v LOS to 0 + perturbations
            std = 2500
            deltas_vlos = self.rng.normal(loc=0.0, scale=std, size=self.ntau[i])
            f = interp.interp1d(self.ltau_nodes[i], deltas_vlos, kind='quadratic', bounds_error=False, fill_value="extrapolate")
            vlos_new = 0 + f(self.ltau[i])

            # Select the depth as the column mass and the depth scale acordingly
            depth = self.atmosRef[i].z
            depth_scale = lw.ScaleType.Geometric

            return depth_scale, depth, T_new, vlos_new, vturb_new, ne


def master_work(nsamples, train, prd_active, savedir, readdir, filename, write_frequency=1, seed=1234):
    """ Function to define the work to do by the master """
    # Calling the Model_generator to read the models and initialice the class
    mg = Model_generator(train, readdir, seed=seed)

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
    vlos_list = [None] * nsamples               # line of sight velocities
    z_list = [None] * nsamples                  # Column mass
    ne_list = [None] * nsamples                 # density of electrons in the atmosphere
    Iwave_list = [None] * nsamples              # Intensity profile of the model
    wave_grid = None                            # wavelength grid the Iwave arrays live on

    success = True

    tasks_status = [0] * nsamples
    last_dump = 0                               # value of pbar.n at the previous dump

    def dump_all():
        """Write every array to disk. Dumps the full lists: the old code sliced [0:task_index],
        which is the index of the last *dispatched* task, so it both trailed the finished work
        and could include unfinished None entries."""
        arrays = {'logdeparture': log_departure_list, 'n_Nat': n_Nat_list, 'T': T_list,
                  'vturb': vturb_list, 'vlos': vlos_list, 'tau': tau_list, 'z': z_list,
                  'ne': ne_list, 'Iwave': Iwave_list, 'wave': wave_grid}
        for name, data in arrays.items():
            with open(savedir + f'{filename}_{name}.pkl', 'wb') as filehandle:
                pickle.dump(data, filehandle)

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
                    if wave_grid is None:
                        # Same for every sample (it depends only on the atomic models), but it was
                        # never stored, which left the Iwave arrays in the database unusable
                        # without re-deriving the grid by hand.
                        wave_grid = dataReceived['wave']
                    log_departure_list[index] = dataReceived['log_departure']
                    n_Nat_list[index] = dataReceived['n_Nat']
                    T_list[index] = dataReceived['T']
                    tau_list[index] = dataReceived['tau']
                    vturb_list[index] = dataReceived['vturb']
                    vlos_list[index] = dataReceived['vlos']
                    z_list[index] = dataReceived['zz']
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

            # Checkpoint the database once every write_frequency *completed* samples. The old
            # condition (pbar.n % write_frequency == 0) was re-evaluated after every MPI message,
            # READY included, so all nine files were rewritten many times over while the counter
            # sat on a multiple -- at this database's size, hours of pointless I/O.
            if pbar.n >= last_dump + write_frequency:
                last_dump = pbar.n
                dump_all()

    # Once finished, dump all the data
    print("Master finishing")
    dump_all()


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
            zz = None
            wave = np.linspace(1074.0, 1085.0, 1100)
            Iwave = wave*0
            success = 1
            # print(f" * WORKER {rank}: starting task {task_index}.", flush=True)

            try:
                # Compute the new atmosphere and solve the NLTE problem and retrieve the solved parameters
                atmos = lw.Atmosphere.make_1d(scale=depth_scale, depthScale=depth, temperature=temperature,
                                              vlos=vlos, vturb=vturb, verbose=False, ne=ne)

                if ne is None:
                    # The reference-atmosphere branch of new_model() hands us ne=None, so the call
                    # above ran a hydrostatic reconstruction and invented *both* ne and nHTot. We
                    # only ever store ne, and any consumer of this database (api.py, and any
                    # inversion built on it) rebuilds the atmosphere by passing that ne back in --
                    # which sends lightweaver down the other EOS branch, deriving nHTot from the
                    # electron pressure instead. The two branches disagree (typically 0.01-0.3 dex
                    # in nHTot, far more for pathological columns), so the departure coefficients
                    # solved on the hydrostatic atmosphere do not belong to the atmosphere the
                    # features reconstruct -- a hidden variable the network cannot see.
                    #
                    # Rebuilding here from the reconstructed ne puts generation on exactly the same
                    # code path as inference, so a stored (T, z, ne, vturb, vlos) column now
                    # determines its own target. The Bifrost branch already supplies ne and so
                    # already took this path; this makes the two consistent. The cost is that the
                    # solved atmosphere is no longer in strict hydrostatic equilibrium -- it is the
                    # EOS's reading of the hydrostatic electron density -- which is the same
                    # compromise the Bifrost columns have always carried.
                    atmos = lw.Atmosphere.make_1d(
                        scale=lw.ScaleType.Geometric,
                        depthScale=np.ascontiguousarray(atmos.z, dtype=np.float64),
                        temperature=np.ascontiguousarray(atmos.temperature, dtype=np.float64),
                        vlos=np.ascontiguousarray(atmos.vlos, dtype=np.float64),
                        vturb=np.ascontiguousarray(atmos.vturb, dtype=np.float64),
                        ne=np.ascontiguousarray(atmos.ne, dtype=np.float64),
                        verbose=False)

                ctx = synth_spectrum(atmos, depthData=True, conserveCharge=False, prd=prd_active)
                # print(f" * WORKER {rank}: finished NLTE task {task_index}.", flush=True)
                tau = atmos.tauRef
                zz = atmos.z
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

                wave = np.asarray(ctx.spect.wavelength)
                Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)

                # If the coefficients are not converged (NaN) or blew up to +/-inf
                # (e.g. a population underflowing to exactly 0 before the log10) set as failure
                if not np.all(np.isfinite(log_departure)):
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
                          'tau': tau, 'zz': zz, 'vlos': vlos, 'vturb': vturb, 'ne': ne, 'success': success,
                          'Iwave': Iwave, 'wave': wave}
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
        parser.add_argument('--seed', '--seed', default=1234, type=int, metavar='SEED',
                            help='seed for the order in which Bifrost columns are consumed and for every '
                                 'perturbation of the reference atmospheres. The train/test partition itself '
                                 'is spatial and seed-independent (see TRAIN_X_FRACTION)')
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

        master_work(parsed['n'], parsed['train'], parsed['prd'], parsed['sav'], parsed['rd'], filename,
                    write_frequency=parsed['f'], seed=parsed['seed'])
    else:
        # Worker processes wait for master to finish initialization
        print(f"Worker {rank}: Waiting for master to finish initialization...", flush=True)
        comm.barrier()
        print(f"Worker {rank}: Starting slave work...", flush=True)
        slave_work(rank)
