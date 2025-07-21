import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Define a smaller 3D grid for visualization (to avoid excessive computation)
grid_size = [10, 10, 10]  # Reduced from [32, 32, 261]
num_nodes = grid_size[0] * grid_size[1] * grid_size[2]

# Build node indexes
nodes_indexes = np.empty((num_nodes, 3))
i = 0
for zpos in tqdm(range(grid_size[2]), desc='Building the indexes'):
    for ypos in range(grid_size[1]):
        for xpos in range(grid_size[0]):
            nodes_indexes[i, :] = np.array([xpos, ypos, zpos])
            i += 1

# Build KDTree and find neighbors within radius 2
tree = KDTree(nodes_indexes, leaf_size=100)
receivers_list = tree.query_radius(nodes_indexes, r=1.75)

# Create sender and receiver arrays
senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
receivers = np.concatenate(receivers_list, axis=0)

# Mask self-edges
mask = senders != receivers
senders = senders[mask]
receivers = receivers[mask]

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
ax.scatter(nodes_indexes[:, 0], nodes_indexes[:, 1], nodes_indexes[:, 2], 
           c='blue', s=10, alpha=0.6, label='Nodes')

# Plot edges (limited to a subset for clarity)
max_edges = 1000  # Limit to avoid clutter
for i in range(min(len(senders), max_edges)):
    sender_pos = nodes_indexes[senders[i]]
    receiver_pos = nodes_indexes[receivers[i]]
    ax.plot([sender_pos[0], receiver_pos[0]],
            [sender_pos[1], receiver_pos[1]],
            [sender_pos[2], receiver_pos[2]], 
            'r-', alpha=0.2)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Graph Visualization (10x10x10 Grid)')

# Adjust view angle for better visibility
ax.view_init(elev=20, azim=45)
plt.show()