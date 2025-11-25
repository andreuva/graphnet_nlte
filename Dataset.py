# DATASET CLASS DEFINITION

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, list_X: list, list_Y: list, radius_neighbors: float, xdim: int, ydim: int,
                 fully_connected: bool = False, pos_file: str = None, split: str = 'train', train_ratio: float = 0.75,
                 logspace_fraction: float = 0.4, nz_linear: int = 30, nz_log: int = 25, epoch_size_fraction: float = 0.2,
                 max_stride: int = 1,
                 random_stride: bool = False,
                 seed: int = 777,
                 device: str = 'cpu'):
        super(EfficientDataset, self).__init__()
        print(f'Dataset.py: Initializing {split} dataset...')
        self.device = device
        self.epoch_size_fraction = epoch_size_fraction
        self.radius = radius_neighbors
        self.xdim = xdim
        self.ydim = ydim
        self.fully_connected = fully_connected
        self.max_stride = max_stride
        self.random_stride = random_stride

        # Store data as numpy arrays to keep them on CPU until needed
        self.features = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_X], axis=1)
        self.targets = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_Y], axis=1)

        self.nz, self.ny, self.nx = list_X[0].shape[:-1]

        grid_pos = np.load(pos_file)
        self.xx, self.yy, self.zz = grid_pos['x'], grid_pos['y'], grid_pos['z']
        nz_original = len(self.zz)

        new_z_indices = np.concatenate([
            np.linspace(0, nz_original*logspace_fraction, nz_linear, endpoint=False),
            np.logspace(np.log10(nz_original*logspace_fraction), np.log10(nz_original-1), nz_log)
        ])
        new_z_indices = np.clip(new_z_indices, 0, nz_original - 1)
        self.zz = np.interp(new_z_indices, np.arange(nz_original), self.zz)

        # Create the physical coordinates grid (x, y, z)
        xgrid, ygrid, zgrid = np.meshgrid(self.xx, self.yy, self.zz, indexing='xy')
        # We need to transpose to match the (z, y, x) structure of the data cubes
        xgrid, ygrid, zgrid = xgrid.T, ygrid.T, zgrid.T
        self.grid_pos = torch.tensor(np.stack([xgrid.ravel(), ygrid.ravel(), zgrid.ravel()], axis=1), dtype=torch.float)

        # If fully connected, create the transform here
        if self.fully_connected:
            print("Warning: Fully connected graphs can be very large and may lead to memory issues.")
            self.radius_transform = RadiusGraph(r=self.radius, loop=False, num_workers=4)

        valid_ix = np.arange(xdim * self.max_stride, self.nx - xdim * self.max_stride)
        valid_iy = np.arange(ydim * self.max_stride, self.ny - ydim * self.max_stride)
        all_indices = [(x, y) for x in valid_ix for y in valid_iy]

        # Shuffle
        np.random.seed(seed)
        np.random.shuffle(all_indices)

        # split into train and test
        x_threshold = int((self.nx - (xdim * self.max_stride)) * train_ratio) + (xdim * self.max_stride)
        y_threshold = int((self.ny - (ydim * self.max_stride)) * train_ratio) + (ydim * self.max_stride)

        if split == 'test':
            self.sample_centers = [(x, y) for x, y in all_indices if x >= x_threshold and y >= y_threshold]
        elif split == 'train':
            self.sample_centers = [(x, y) for x, y in all_indices if x < x_threshold or y < y_threshold]
        elif split == 'full':
            self.sample_centers = [(x, y) for x, y in all_indices]
        else:
            raise ValueError("split must be 'train' or 'test'")

        print(f'Dataset.py:  {split.capitalize()} dataset created. Total samples: {len(self.sample_centers)}')
        print(f'Dataset.py:  Features shape: {self.features.shape}, Targets shape: {self.targets.shape}')
        print(f'Dataset.py:  Test region: x ∈ [{x_threshold}, {self.nx - xdim * self.max_stride}), y ∈ [{y_threshold}, {self.ny - ydim * self.max_stride})')
        print(f'Dataset.py:  Split ratio: {len(self.sample_centers) / len(all_indices) * 100:.2f}% of all valid samples')

    def __len__(self):
        return int(len(self.sample_centers)*self.epoch_size_fraction)

    def grid_to_graph_manual(self, grid_points, values=None, targets=None, r=1.5, xpos=None, ypos=None):
        if values is None:
            values = grid_points
        if targets is None:
            targets = grid_points

        dist_matrix = torch.cdist(grid_points, grid_points)
        edge_indices_tuple = torch.where((dist_matrix > 0) & (dist_matrix <= r))
        edge_index = torch.stack(edge_indices_tuple, dim=0)

        graph_data = Data(x=values, pos=grid_points, y=targets, edge_index=edge_index)

        if xpos is not None and ypos is not None:
            central_nodes = torch.where(
                (grid_points[:, 2] == xpos) & (grid_points[:, 1] == ypos)
            )[0]

            if len(central_nodes) > 0:
                edge_mask = torch.from_numpy(
                    np.isin(graph_data.edge_index[0, :], central_nodes) |
                    np.isin(graph_data.edge_index[1, :], central_nodes)
                )
                graph_data.edge_index = graph_data.edge_index[:, edge_mask]

        return graph_data

    def __getitem__(self, index):
        ix, iy = self.sample_centers[index]

        # A stride of 1 is the original contiguous block.
        if self.random_stride:
            x_stride = np.random.randint(1, self.max_stride + 1)
            y_stride = np.random.randint(1, self.max_stride + 1)
        else:
            x_stride = self.max_stride
            y_stride = self.max_stride

        # Create offsets from the center, multiply by stride
        x_offsets = np.arange(-self.xdim, self.xdim + 1) * x_stride
        y_offsets = np.arange(-self.ydim, self.ydim + 1) * y_stride

        # Create the new sparse ranges, ensuring they don't go out of bounds
        x_range = np.clip(ix + x_offsets, 0, self.nx - 1)
        y_range = np.clip(iy + y_offsets, 0, self.ny - 1)

        k_range = np.arange(self.nz)

        # Create a grid of coordinates for the sub-volume (as indices)
        kv, yv, xv = np.meshgrid(k_range, y_range, x_range, indexing='ij')

        # Flatten and stack to get node positions (indices), used for connectivity
        node_pos_indices = torch.tensor(np.stack([kv.ravel(), yv.ravel(), xv.ravel()], axis=1), dtype=torch.float)
        # Calculate flat indices from the coordinate grid to slice the full data arrays
        flat_indices = (kv.ravel() * self.ny * self.nx + yv.ravel() * self.nx + xv.ravel())

        # The mask is True for nodes where the (y, x) coordinates match the center (iy, ix)
        central_mask = (node_pos_indices[:, 1] == iy) & (node_pos_indices[:, 2] == ix)

        # Get physical positions, features, targets, and add z-position feature
        sub_physical_pos = self.grid_pos[flat_indices]

        # Get the features and targets for this sub-grid
        node_features = torch.tensor(self.features[flat_indices], dtype=torch.float)
        node_targets = torch.tensor(self.targets[flat_indices], dtype=torch.float)

        # self.grid_pos is (x, y, z), so z is at index 2
        z_coords = sub_physical_pos[:, 2].unsqueeze(1)
        node_features = torch.cat([node_features, z_coords], dim=1)

        if self.fully_connected:
            data = Data(x=node_features, pos=node_pos_indices, y=node_targets)
            graph_data = self.radius_transform(data)
        else:
            graph_data = self.grid_to_graph_manual(
                # node_pos_indices, node_features, node_targets, r=self.radius, xpos=ix, ypos=iy
                node_pos_indices, node_features, node_targets, r=self.radius*(x_stride + y_stride)/3, xpos=ix, ypos=iy
            )

        graph_data.central_mask = central_mask

        # Calculate edge attributes using PHYSICAL distances
        if graph_data.edge_index.numel() > 0:
            row, col = graph_data.edge_index
            # Calculate Euclidean distance based on physical positions
            edge_vectors_physical = sub_physical_pos[row] - sub_physical_pos[col]
            graph_data.edge_attr = edge_vectors_physical.norm(dim=1).unsqueeze(1)
        else:
            # Handle the case of no edges
            graph_data.edge_attr = torch.empty((0, 1), dtype=torch.float)

        graph_data.u = torch.zeros((1, 1), dtype=torch.float)
        return graph_data