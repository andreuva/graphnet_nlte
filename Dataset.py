import torch
import pickle
from tqdm import tqdm
import os
import numpy as np
import torch_geometric.data

# Feature normalization statistics (Z-score: (x - mean) / std), computed on 2026-10-02 with
# normalization_compute.py over every depth point of the full data_1d_si_v4/train_* split
# (996042 columns, 50.3% of them 211-point Bifrost columns, 1.49e8 depth points). Recompute if
# the training set changes materially (different atom/species, different atmosphere mix, etc.);
# every checkpoint stores the values it was trained with, so retuning these never affects an
# existing checkpoint. Previous values (data_1d_si, 493391 columns, 20% Bifrost; used by every
# run started before 2026-10-02): T_log10 4.10228/0.600127, z 1.30265e6/1.09178e6, tau_log10
# -7.75738/4.37718, ne_log10 17.5785/2.35518, vturb_km 2.88109/6.14624, vlos_km
# -0.603487/4.86515, delta_z 17706.5.
NORM_STATS = {
    'T_log10': {'mean': 4.13, 'std': 0.632214},
    'z': {'mean': 1.3246e6, 'std': 1.11838e6},       # height in meters
    'tau_log10': {'mean': -7.89841, 'std': 4.06044},
    'ne_log10': {'mean': 17.511, 'std': 2.41168},
    'vturb_km': {'mean': 8.17848, 'std': 8.44839},   # vturb in km/s (scaled by 1e3)
    'vlos_km': {'mean': -0.838815, 'std': 4.05366},  # vlos in km/s (scaled by 1e3)
    # RMS of the z difference between adjacent depth points (edge feature). The feature is built
    # for both directions of every edge, so it has zero mean and the RMS makes it unit variance;
    # the previous value was the plain std of the one-directional differences.
    'delta_z': {'std': 26053.7},
}

# Levels and depths whose population falls below this fraction of the species total,
# log10(n_i / n_Total), are dropped from the loss. They are physically inert -- they change the
# emergent Si I 1083 nm profile by less than a part in 1e6 -- but they are numerically loud:
# roughly 16% of all target values sit on the +-10 clip plateau at |y| = 2 after scaling, while
# the entire line-forming region lives inside |y| <= 0.2, so an unmasked MSE spends most of its
# gradient there. At -9 this masks 20% of the points and removes 83% of the clipped plateau.
# Requires the *_n_Nat.pkl files; without them nothing is masked.
NEGLIGIBLE_LOG_N_OVER_NTOT = -9.0

# Long-range "ladder" edges on top of the nearest-neighbour chain. For each level k, hub nodes
# are placed at the k+1 points j*(N-1)/k of an N-point column and consecutive hubs are linked:
# with (2, 3, 10) a 100-point column gets 0-50-99, 0-33-66-99 and 0-10-20-...-99. On the chain
# alone information travels one point per message-passing step, so with 128 steps the top of a
# 211-point Bifrost column never sees the photosphere whose radiation field sets its
# populations; along the ladder any two points are a handful of steps apart, while all but a
# few nodes keep their plain chain neighbourhood (linking every node to far-away points would
# blur the local transfer). The levels a checkpoint was trained with are stored in it as
# 'edge_ladder_levels' (Formal.py), so Formal.test and api._build_graph rebuild exactly the
# graph the checkpoint expects; () is the plain chain of older checkpoints.
EDGE_LADDER_LEVELS = (2, 3, 10)


def build_edge_index(num_nodes, ladder_levels=EDGE_LADDER_LEVELS):
    """
    Chain graph of a column (node j <-> j-1 and j <-> j+1) plus the ladder rungs of
    `ladder_levels` (see EDGE_LADDER_LEVELS), as a (2, n_edges) int64 array of
    [senders, receivers] with every edge in both directions. Kept identical to
    api._build_edge_index.
    """
    pairs = {(j, j + 1) for j in range(num_nodes - 1)}
    for k in ladder_levels:
        hubs = np.unique(np.round(np.linspace(0, num_nodes - 1, k + 1)).astype(int))
        pairs.update((int(a), int(b)) for a, b in zip(hubs[:-1], hubs[1:]))
    pairs = sorted(pairs)
    a = np.array([p[0] for p in pairs], dtype=np.int64)
    b = np.array([p[1] for p in pairs], dtype=np.int64)
    return np.stack([np.concatenate([a, b]), np.concatenate([b, a])])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, hyperparameters, datadir='../data/', prefix='train', norm_stats=None,
                 edge_ladder_levels=EDGE_LADDER_LEVELS, edge_span_scaled=True, node_drop=0.0):
        """
        Dataset for the depth stratification

        norm_stats : dict, optional
            Normalization constants to build the features with. Defaults to the module-level
            NORM_STATS; pass a checkpoint's embedded 'norm_stats' when reproducing the exact
            inputs an older checkpoint was trained with.
        edge_ladder_levels : tuple of int, optional
            Ladder levels of long-range edges added to the chain, see EDGE_LADDER_LEVELS. Pass
            a checkpoint's embedded 'edge_ladder_levels' when evaluating it.
        edge_span_scaled : bool, optional
            With edge_input_size 2, divide the delta-z edge feature by the number of depth points
            the edge spans, i.e. use the mean spacing along the edge (see _edge_attr). True for
            new runs; pass a checkpoint's embedded 'edge_span_scaled' when evaluating it (False
            when absent: runs started before 2026-10-02 saw the raw delta z on the rungs).
        node_drop : float, optional
            Training-time augmentation. Each time a column is fetched, a random fraction of its
            interior depth points, uniform in [0, node_drop], is dropped and the graph is rebuilt
            on the remaining points (the two ends always stay). The departure coefficients are
            per depth point, so the targets of the surviving points are unchanged; the network
            just sees the same atmosphere at a different z resolution. 0 (default) disables it;
            keep it 0 for validation and test.
        """
        super(Dataset, self).__init__()

        ns = NORM_STATS if norm_stats is None else norm_stats
        self.ns = ns
        self.edge_ladder_levels = tuple(edge_ladder_levels)
        self.edge_span_scaled = bool(edge_span_scaled)
        self.node_drop = float(node_drop)
        self.edge_input_size = hyperparameters['edge_input_size']

        # Read the training database
        with open(datadir + prefix + '_tau.pkl', 'rb') as filehandle:
            self.tau_all = pickle.load(filehandle)

        with open(datadir + prefix + '_vturb.pkl', 'rb') as filehandle:
            self.vturb_all = pickle.load(filehandle)

        with open(datadir + prefix + '_vlos.pkl', 'rb') as filehandle:
            self.vlos_all = pickle.load(filehandle)

        with open(datadir + prefix + '_T.pkl', 'rb') as filehandle:
            self.T_all = pickle.load(filehandle)

        with open(datadir + prefix + '_logdeparture.pkl', 'rb') as filehandle:
            self.dep_all = pickle.load(filehandle)

        self.n_Nat_activated = False
        if os.path.isfile(datadir + prefix + '_n_Nat.pkl'):
            self.n_Nat_activated = True
            with open(datadir + prefix + '_n_Nat.pkl', 'rb') as filehandle:
                self.n_Nat = pickle.load(filehandle)

        self.ne_activated = False
        if os.path.isfile(datadir + prefix + '_ne.pkl'):
            self.ne_activated = True
            with open(datadir + prefix + '_ne.pkl', 'rb') as filehandle:
                self.ne = pickle.load(filehandle)

        self.z_activated = False
        if os.path.isfile(datadir + prefix + '_z.pkl'):
            self.z_activated = True
            with open(datadir + prefix + '_z.pkl', 'rb') as filehandle:
                self.z_all = pickle.load(filehandle)

        # Now we need to define the graphs for each one of the computed models
        # The graph will connect all points at certain distance. We define this distance
        # as integer indices, so that we make sure that nodes are connected to the neighbors
        self.n_training = len(self.T_all)

        # Initialize the graph information
        self.edge_index = [None] * self.n_training
        self.nodes = [None] * self.n_training
        self.edges = [None] * self.n_training
        self.u = [None] * self.n_training
        self.target = [None] * self.n_training
        self.mask = [None] * self.n_training

        # The chain connectivity depends only on the number of depth points, so one edge_index
        # tensor is shared by every column of the same length (int64, 2 x 2(N-1) per column).
        edge_index_cache = {}

        # Loop over all training examples
        for i in tqdm(range(self.n_training), "Preparing graphs..."):

            num_nodes = len(self.tau_all[i])

            # Chain plus ladder edges (same edge set as api._build_graph, see build_edge_index)
            if num_nodes not in edge_index_cache:
                edge_index_cache[num_nodes] = torch.tensor(
                    build_edge_index(num_nodes, self.edge_ladder_levels), dtype=torch.long)
            self.edge_index[i] = edge_index_cache[num_nodes]

            # Define normalized node features (mean=0, std=1)
            node_input_size = hyperparameters['node_input_size']
            self.nodes[i] = np.zeros((num_nodes, node_input_size))

            # Feature 0: log10(T)
            self.nodes[i][:, 0] = (np.log10(self.T_all[i]) - ns['T_log10']['mean']) / ns['T_log10']['std']

            # Feature 1: Height z or log10(tau)
            if node_input_size > 1:
                if self.z_activated:
                    self.nodes[i][:, 1] = (self.z_all[i] - ns['z']['mean']) / ns['z']['std']
                else:
                    self.nodes[i][:, 1] = (np.log10(self.tau_all[i]) - ns['tau_log10']['mean']) / ns['tau_log10']['std']

            # Feature 2: log10(ne)
            if node_input_size > 2 and self.ne_activated:
                self.nodes[i][:, 2] = (np.log10(self.ne[i]) - ns['ne_log10']['mean']) / ns['ne_log10']['std']

            # Feature 3: vturb [km/s]
            if node_input_size > 3:
                self.nodes[i][:, 3] = (self.vturb_all[i] / 1e3 - ns['vturb_km']['mean']) / ns['vturb_km']['std']

            # Feature 4: vlos [km/s]
            if node_input_size > 4:
                self.nodes[i][:, 4] = (self.vlos_all[i] / 1e3 - ns['vlos_km']['mean']) / ns['vlos_km']['std']

            # Define normalized edge features
            self.edges[i] = self._edge_attr(i, self.edge_index[i].numpy())

            # We don't use at the moment any global property of the graph, so we set it to zero.
            self.u[i] = np.zeros((1, 1))
            # self.u[i][0, :] = np.array([np.log10(self.eps_all[i][0, 0]), np.log10(self.ratio_all[i][0, 0])], dtype=np.float32)

            # We use the log10(departure coeff) as output, divided by 5 to make it closer to 1.
            # log10(b) is clipped to +-10 first: beyond that range b is dominated by a Saha/LTE
            # collapse of the reference population (nStar -> 0, mostly in the transition region/
            # corona) rather than by the actual level population, which by then is astrophysically
            # negligible (n/Ntotal below ~1e-9) -- so the true value doesn't matter, and letting it
            # through unclipped just adds noisy outliers to the MSE loss. In case a NaN or Inf is
            # still found (non-converged sample slipping through), we make it zero.
            self.target[i] = np.nan_to_num(np.clip(self.dep_all[i][:, :].T, -10.0, 10.0) / 5.0)

            # Per (depth, level) weight of 1/0 for the loss. A NaN in n_Nat compares False and is
            # therefore masked out, which is what we want: it means the population underflowed to
            # zero, i.e. the level is empty.
            if self.n_Nat_activated:
                with np.errstate(invalid='ignore'):
                    self.mask[i] = (self.n_Nat[i][:, :].T >= NEGLIGIBLE_LOG_N_OVER_NTOT)
            else:
                self.mask[i] = np.ones_like(self.target[i], dtype=bool)

            # Finally, all information is transformed to float32 tensors
            self.nodes[i] = torch.tensor(self.nodes[i].astype('float32'))
            self.edges[i] = torch.tensor(self.edges[i].astype('float32'))
            self.u[i] = torch.tensor(self.u[i].astype('float32'))
            self.target[i] = torch.tensor(self.target[i].astype('float32'))
            self.mask[i] = torch.tensor(self.mask[i])

        # The raw n_Nat arrays are only needed to build the masks above, and they are as large as
        # the departure coefficients themselves (8.7 GB for the training split). The raw
        # departure coefficients are fully represented by the float32 targets (see __call__).
        self.n_Nat = None
        self.dep_all = None

    def _edge_attr(self, i, edge_index, keep=None):
        """
        Normalized edge features of column i for a (2, n_edges) edge_index. `keep`, if given, is
        the sorted array of the column's depth points that edge_index refers to (node dropping in
        __getitem__); otherwise the edges index the full column.
        """
        ns = self.ns
        src, dst = edge_index[0], edge_index[1]
        if keep is not None:
            src, dst = keep[src], keep[dst]
        edges = np.zeros((edge_index.shape[1], self.edge_input_size))

        if self.edge_input_size == 2:
            # Feature 1: log10 of the number of depth points the edge spans in the current graph
            # (between survivors under node dropping): exactly 0 on the chain, ~1-2 on the ladder
            # rungs, so the edge encoder can tell a rung from a chain edge by an order-one input
            # at any resolution. Feature 0: delta z in units of the RMS adjacent spacing and,
            # with edge_span_scaled, divided by that span, i.e. the mean spacing along the edge:
            # unchanged on the chain (span 1), order one on the rungs instead of ~100, and the
            # full delta z is still feature 0 times 10**feature 1. Runs started before 2026-10-02
            # saw the undivided value on the rungs; their checkpoints carry no 'edge_span_scaled'
            # key and are evaluated with edge_span_scaled=False. (Before 2026-09-30 the second
            # feature was delta log10 tau, which inference cannot compute.) Same construction as
            # api._build_graph.
            if not self.z_activated:
                raise ValueError("No z data available for edge features")
            span = np.abs(edge_index[1] - edge_index[0])
            edges[:, 0] = (self.z_all[i][src] - self.z_all[i][dst]) / ns['delta_z']['std']
            if self.edge_span_scaled:
                edges[:, 0] /= span
            edges[:, 1] = np.log10(span)
        elif self.edge_input_size == 1:
            if self.z_activated:
                edges[:, 0] = (self.z_all[i][src] - self.z_all[i][dst]) / ns['delta_z']['std']
            else:
                tau0 = np.log10(self.tau_all[i][src])
                tau1 = np.log10(self.tau_all[i][dst])
                edges[:, 0] = (tau0 - tau1) / ns['tau_log10']['std']
        else:
            raise ValueError("Incompatible edge input size")
        return edges

    def __getitem__(self, index):

        # When we are asked to return the information of a graph, we encode
        # it in a Data class. Batches in graphs work slightly different than
        # in more classical situations. Since we have the connectivity of each
        # graph, batches are built by generating a big graph containing all
        # graphs of the batch.
        node = self.nodes[index]
        edge_attr = self.edges[index]
        target = self.target[index]
        u = self.u[index]
        edge_index = self.edge_index[index]
        mask = self.mask[index]

        if self.node_drop > 0:
            # Drop a random fraction, uniform in [0, node_drop], of the interior depth points and
            # rebuild the graph on the survivors; the two ends stay (they bound the column). Uses
            # torch's RNG, which the DataLoader seeds per worker and per epoch from its own
            # generator, so --seed still fixes the augmentation.
            n = node.shape[0]
            n_drop = int(torch.rand(()).item() * self.node_drop * (n - 2))
            if n_drop > 0:
                drop = torch.randperm(n - 2)[:n_drop] + 1
                keep = np.setdiff1d(np.arange(n), drop.numpy())
                new_edge_index = build_edge_index(len(keep), self.edge_ladder_levels)
                edge_attr = torch.tensor(self._edge_attr(index, new_edge_index, keep).astype('float32'))
                edge_index = torch.tensor(new_edge_index, dtype=torch.long)
                keep = torch.from_numpy(keep)
                node, target, mask = node[keep], target[keep], mask[keep]

        data = torch_geometric.data.Data(x=node, edge_index=edge_index, edge_attr=edge_attr, y=target, u=u,
                                         mask=mask)

        return data

    def __len__(self):
        return self.n_training

    def __call__(self, index):
        # log10(b) in the stored (n_levels, n_depth) layout, recovered from the scaled target
        # (clipped to +-10, which is what every consumer of this database uses anyway).
        log_dep = self.target[index].numpy().T * 5.0
        return self.T_all[index], self.z_all[index], self.ne[index], self.vturb_all[index], self.vlos_all[index], self.u[index], log_dep
