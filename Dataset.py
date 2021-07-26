import torch
import pickle
from tqdm import tqdm
import os
import numpy as np
from sklearn import neighbors
import torch_geometric.data


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

        self.cmass_activated = False
        if os.path.isfile(datadir + prefix + '_cmass.pkl'):
            self.cmass_activated = True
            with open(datadir + prefix + '_cmass.pkl', 'rb') as filehandle:
                self.cmass_all = pickle.load(filehandle)

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
        for i in tqdm(range(self.n_training)):

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

            # Now define the nodes. For the moment we use only one quantity, the log10(T)
            node_input_size = hyperparameters['node_input_size']
            self.nodes[i] = np.zeros((num_nodes, node_input_size))
            self.nodes[i][:, 0] = np.log10(self.T_all[i])
            if node_input_size > 1:
                self.nodes[i][:, 1] = np.log10(self.tau_all[i])
            if node_input_size > 2 and self.ne_activated:
                self.nodes[i][:, 2] = np.log10(self.ne[i])
            if node_input_size > 3:
                self.nodes[i][:, 3] = self.vturb_all[i]/1e3
            if node_input_size > 4:
                self.nodes[i][:, 4] = self.vlos_all[i]/1e3

            # We use two quantities for the information encoded on the edges: log(column mass) and log(tau)
            edge_input_size = hyperparameters['edge_input_size']
            self.edges[i] = np.zeros((n_edges, edge_input_size))

            if edge_input_size == 2:
                cmass_0 = np.log10(self.cmass_all[i][self.edge_index[i][0, :]])
                cmass_1 = np.log10(self.cmass_all[i][self.edge_index[i][1, :]])
                self.edges[i][:, 0] = (cmass_0 - cmass_1)

                tau0 = np.log10(self.tau_all[i][self.edge_index[i][0, :]])
                tau1 = np.log10(self.tau_all[i][self.edge_index[i][1, :]])
                self.edges[i][:, 1] = (tau0 - tau1)
            elif edge_input_size == 1:
                tau0 = np.log10(self.tau_all[i][self.edge_index[i][0, :]])
                tau1 = np.log10(self.tau_all[i][self.edge_index[i][1, :]])
                self.edges[i][:, 0] = (tau0 - tau1)
            else:
                raise ValueError("Incompatible edge input size")

            # We don't use at the moment any global property of the graph, so we set it to zero.
            self.u[i] = np.zeros((1, 1))
            # self.u[i][0, :] = np.array([np.log10(self.eps_all[i][0, 0]), np.log10(self.ratio_all[i][0, 0])], dtype=np.float32)

            # We use the log10(departure coeff) as output, divided by 5 to make it closer to 1. In case a NaN is found, we
            # make them equal to zero
            self.target[i] = np.nan_to_num(self.dep_all[i][:, :].T / 5.0)

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
        return self.cmass_all[index], self.tau_all[index], self.vturb_all[index], self.vlos_all[index], self.T_all[index], self.u[index], self.dep_all[index]
