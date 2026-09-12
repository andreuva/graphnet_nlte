import torch
import pickle
from tqdm import tqdm
import os
import numpy as np
from sklearn import neighbors
import torch_geometric.data

# Feature normalization statistics (Z-score: (x - mean) / std), computed over the full
# data_1d_si/train_* set (493391 columns, ~6.8e7 depth points). Recompute if the training
# set changes materially (different atom/species, different atmosphere mix, etc.).
NORM_STATS = {
    'T_log10': {'mean': 4.10228, 'std': 0.600127},
    'z': {'mean': 1.30265e6, 'std': 1.09178e6},      # height in meters
    'tau_log10': {'mean': -7.75738, 'std': 4.37718},
    'ne_log10': {'mean': 17.5785, 'std': 2.35518},
    'vturb_km': {'mean': 2.88109, 'std': 6.14624},   # vturb in km/s (scaled by 1e3)
    'vlos_km': {'mean': -0.603487, 'std': 4.86515},  # vlos in km/s (scaled by 1e3)
    'delta_z': {'std': 17706.5},                     # std of z differences between adjacent nodes (edge feature)
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, hyperparameters, datadir='../data/', prefix='train'):
        """
        Dataset for the depth stratification
        """
        super(Dataset, self).__init__()

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

        # Loop over all training examples
        for i in tqdm(range(self.n_training), "Preparing graphs..."):

            num_nodes = len(self.tau_all[i])
            index_tau = np.zeros((num_nodes, 1))

            index_tau[:, 0] = np.arange(num_nodes)

            # Build the KDTree
            self.tree = neighbors.KDTree(index_tau)

            # Get neighbors
            receivers_list = self.tree.query_radius(index_tau, r=1)

            senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
            receivers = np.concatenate(receivers_list, axis=0)

            # Mask self edges
            mask = senders != receivers

            # Transform senders and receivers to tensors
            senders = torch.tensor(senders[mask].astype('long'))
            receivers = torch.tensor(receivers[mask].astype('long'))

            # Define the graph for this model by using the sender/receiver information
            self.edge_index[i] = torch.cat([senders[None, :], receivers[None, :]], dim=0)

            n_edges = self.edge_index[i].shape[1]

            # Define normalized node features (mean=0, std=1)
            node_input_size = hyperparameters['node_input_size']
            self.nodes[i] = np.zeros((num_nodes, node_input_size))

            # Feature 0: log10(T)
            self.nodes[i][:, 0] = (np.log10(self.T_all[i]) - NORM_STATS['T_log10']['mean']) / NORM_STATS['T_log10']['std']

            # Feature 1: Height z or log10(tau)
            if node_input_size > 1:
                if self.z_activated:
                    self.nodes[i][:, 1] = (self.z_all[i] - NORM_STATS['z']['mean']) / NORM_STATS['z']['std']
                else:
                    self.nodes[i][:, 1] = (np.log10(self.tau_all[i]) - NORM_STATS['tau_log10']['mean']) / NORM_STATS['tau_log10']['std']

            # Feature 2: log10(ne)
            if node_input_size > 2 and self.ne_activated:
                self.nodes[i][:, 2] = (np.log10(self.ne[i]) - NORM_STATS['ne_log10']['mean']) / NORM_STATS['ne_log10']['std']

            # Feature 3: vturb [km/s]
            if node_input_size > 3:
                self.nodes[i][:, 3] = (self.vturb_all[i] / 1e3 - NORM_STATS['vturb_km']['mean']) / NORM_STATS['vturb_km']['std']

            # Feature 4: vlos [km/s]
            if node_input_size > 4:
                self.nodes[i][:, 4] = (self.vlos_all[i] / 1e3 - NORM_STATS['vlos_km']['mean']) / NORM_STATS['vlos_km']['std']

            # Define normalized edge features
            edge_input_size = hyperparameters['edge_input_size']
            self.edges[i] = np.zeros((n_edges, edge_input_size))

            if edge_input_size == 2:
                if not self.z_activated:
                    raise ValueError("No z data available for edge features")
                else:
                    z_0 = self.z_all[i][self.edge_index[i][0, :]]
                    z_1 = self.z_all[i][self.edge_index[i][1, :]]
                    self.edges[i][:, 0] = (z_0 - z_1) / NORM_STATS['delta_z']['std']

                tau0 = np.log10(self.tau_all[i][self.edge_index[i][0, :]])
                tau1 = np.log10(self.tau_all[i][self.edge_index[i][1, :]])
                self.edges[i][:, 1] = (tau0 - tau1) / NORM_STATS['tau_log10']['std']
            elif edge_input_size == 1:
                if self.z_activated:
                    z_0 = self.z_all[i][self.edge_index[i][0, :]]
                    z_1 = self.z_all[i][self.edge_index[i][1, :]]
                    self.edges[i][:, 0] = (z_0 - z_1) / NORM_STATS['delta_z']['std']
                else:
                    tau0 = np.log10(self.tau_all[i][self.edge_index[i][0, :]])
                    tau1 = np.log10(self.tau_all[i][self.edge_index[i][1, :]])
                    self.edges[i][:, 0] = (tau0 - tau1) / NORM_STATS['tau_log10']['std']
            else:
                raise ValueError("Incompatible edge input size")

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

            # Finally, all information is transformed to float32 tensors
            self.nodes[i] = torch.tensor(self.nodes[i].astype('float32'))
            self.edges[i] = torch.tensor(self.edges[i].astype('float32'))
            self.u[i] = torch.tensor(self.u[i].astype('float32'))
            self.target[i] = torch.tensor(self.target[i].astype('float32'))

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

        data = torch_geometric.data.Data(x=node, edge_index=edge_index, edge_attr=edge_attr, y=target, u=u)

        return data

    def __len__(self):
        return self.n_training

    def __call__(self, index):
        return self.T_all[index], self.z_all[index], self.ne[index], self.vturb_all[index], self.vlos_all[index], self.u[index], self.dep_all[index]
