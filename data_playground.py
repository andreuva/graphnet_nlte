###############################################################################################
# BUILD THE GRAPH TO TEST THE 3D
###############################################################################################
from platform import node
from sklearn import neighbors
import numpy as np
import torch
from tqdm import tqdm

data = {}
data['head'] = {}
data['head']['nodes0'] = [504, 504, 261]

num_nodes = data['head']['nodes0'][0]*data['head']['nodes0'][1]*data['head']['nodes0'][2]
print(f'Number of nodes: {num_nodes}')

# Build the indexes of the nodes
nodes_indexes = np.empty((num_nodes,3))
i = 0
for zpos in tqdm(range(data['head']['nodes0'][2]), desc='Building the indexes'):
    for ypos in range(data['head']['nodes0'][1]):
        for xpos in range(data['head']['nodes0'][0]):
            nodes_indexes[i,:] = np.array([zpos,ypos,xpos])
            i += 1

# Build the KDtree
tree = neighbors.KDTree(nodes_indexes, leaf_size=1000)

# Get neighbors
receivers_list = tree.query_radius(nodes_indexes, r=1)

senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
receivers = np.concatenate(receivers_list, axis=0)

# Mask self edges
mask = senders != receivers

# Transform senders and receivers to tensors
senders = torch.tensor(senders[mask].astype('long'))
receivers = torch.tensor(receivers[mask].astype('long'))

# Define the graph for this model by using the sender/receiver information
edge_index = torch.cat([senders[None, :], receivers[None, :]], dim=0)

n_edges = edge_index.shape[1]


# Now define the nodes
node_input_size = 5
nodes = np.zeros((num_nodes, node_input_size))

# nodes[:, 0] = np.log10(data['nodes']['temp'].reshape(-1))
# nodes[i][:, 1] = np.log10(tau_all[i])
# nodes[i][:, 2] = np.log10(ne[i])
# nodes[i][:, 3] = vturb_all[i]/1e3
# nodes[i][:, 4] = vlos_all[i]/1e3

# We use one quantities for the information encoded on the edges: log(tau)
edge_input_size = 1
edges = np.zeros((n_edges, edge_input_size))

# tau0 = np.log10(tau_all[edge_index[i][0, :]])
# tau1 = np.log10(tau_all[edge_index[i][1, :]])
# edges[i][:, 0] = (tau0 - tau1)

# We don't use at the moment any global property of the graph, so we set it to zero.
u = np.zeros((1, 1))

# We use the log10(departure coeff) as output, divided by 5 to make it closer to 1. In case a NaN is found, we
# make them equal to zero
# target = np.nan_to_num(dep_all[i][:, :].T / 5.0)

# Finally, all information is transformed to float32 tensors
nodes = torch.tensor(nodes.astype('float32'))
edges = torch.tensor(edges.astype('float32'))
u = torch.tensor(u.astype('float32'))
# target = torch.tensor(target.astype('float32'))
