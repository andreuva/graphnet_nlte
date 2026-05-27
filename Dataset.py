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

        # Convert everything directly to PyTorch tensors to avoid runtime conversions
        self.features = torch.tensor(np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_X], axis=1), dtype=torch.float32)
        self.targets = torch.tensor(np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_Y], axis=1), dtype=torch.float32)

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
        self.grid_pos = torch.tensor(np.stack([xgrid.ravel(), ygrid.ravel(), zgrid.ravel()], axis=1), dtype=torch.float32)

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
            self.x0, self.y0 = x_threshold, y_threshold
            self.x1, self.y1 = self.nx - xdim * self.max_stride, self.ny - ydim * self.max_stride
            self.sample_centers = [(x, y) for x, y in all_indices if x >= x_threshold and y >= y_threshold]
        elif split == 'train':
            self.x0, self.y0 = xdim * self.max_stride, ydim * self.max_stride
            self.x1, self.y1 = x_threshold, y_threshold
            self.sample_centers = [(x, y) for x, y in all_indices if x < x_threshold or y < y_threshold]
        elif split == 'full':
            self.x0, self.y0 = xdim * self.max_stride, ydim * self.max_stride
            self.x1, self.y1 = self.nx - xdim * self.max_stride, self.ny - ydim * self.max_stride
            self.sample_centers = [(x, y) for x, y in all_indices]
        else:
            raise ValueError("split must be 'train' or 'test'")

        # PRE-ALLOCATIONS to save CPU cycles
        self.static_u = torch.zeros((1, 1), dtype=torch.float32)
        self._precompute_graph_templates()

        print(f'Dataset.py:  {split.capitalize()} dataset created. Total samples: {len(self.sample_centers)}')
        print(f'Dataset.py:  Features shape: {self.features.shape}, Targets shape: {self.targets.shape}')
        print(f'Dataset.py:  Region: x ∈ [{self.x0}, {self.x1}), y ∈ [{self.y0}, {self.y1})')
        print(f'Dataset.py:  Split ratio: {len(self.sample_centers) / len(all_indices) * 100:.2f}% of all valid samples')
        print(f'Dataset.py:  Using max stride: {self.max_stride}, Random stride: {self.random_stride}')

    def _precompute_graph_templates(self):
        """
        Calculates all possible edge connections and flat index offsets ONCE.
        This removes O(N^2) cdist calculations from the __getitem__ loop.
        """
        self.templates = {}
        k_range = np.arange(self.nz)
        
        for xs in range(1, self.max_stride + 1):
            for ys in range(1, self.max_stride + 1):
                x_offs = np.arange(-self.xdim, self.xdim + 1) * xs
                y_offs = np.arange(-self.ydim, self.ydim + 1) * ys
                
                kv, yv, xv = np.meshgrid(k_range, y_offs, x_offs, indexing='ij')
                
                # Precompute the relative flatten offset for 1D slicing
                relative_flat = kv.ravel() * self.ny * self.nx + yv.ravel() * self.nx + xv.ravel()
                
                # Mock indices to calculate fixed graph topology
                node_pos_mock = torch.tensor(np.stack([kv.ravel(), yv.ravel(), xv.ravel()], axis=1), dtype=torch.float32)
                
                central_mask = (yv.ravel() == 0) & (xv.ravel() == 0)
                
                if not self.fully_connected:
                    r_eff = self.radius * (xs + ys) / 3
                    dist_matrix = torch.cdist(node_pos_mock, node_pos_mock)
                    edge_indices_tuple = torch.where((dist_matrix > 0) & (dist_matrix <= r_eff))
                    edge_index = torch.stack(edge_indices_tuple, dim=0)
                    
                    central_nodes = torch.where(torch.tensor(central_mask))[0]
                    if len(central_nodes) > 0:
                        edge_mask = torch.isin(edge_index[0], central_nodes) | torch.isin(edge_index[1], central_nodes)
                        edge_index = edge_index[:, edge_mask]
                else:
                    edge_index = None # Will be handled by RadiusGraph on the fly
                    
                self.templates[(xs, ys)] = {
                    'relative_flat': torch.tensor(relative_flat, dtype=torch.long),
                    'edge_index': edge_index,
                    'central_mask': torch.tensor(central_mask, dtype=torch.bool),
                    'relative_pos': node_pos_mock
                }

    def __len__(self):
        return int(len(self.sample_centers)*self.epoch_size_fraction)

    def __getitem__(self, index):
        ix, iy = self.sample_centers[index]

        # A stride of 1 is the original contiguous block.
        if self.random_stride:
            x_stride = np.random.randint(1, self.max_stride + 1)
            y_stride = np.random.randint(1, self.max_stride + 1)
        else:
            x_stride = self.max_stride
            y_stride = self.max_stride

        # Fetch the precomputed topology template
        template = self.templates[(x_stride, y_stride)]
        
        # Shift the precomputed flattened indices to the target absolute center
        center_flat = iy * self.nx + ix
        flat_indices = template['relative_flat'] + center_flat

        # Direct tensor indexing (avoids data copies and from_numpy conversions)
        node_features = self.features[flat_indices]
        node_targets = self.targets[flat_indices]

        sub_physical_pos = self.grid_pos[flat_indices]
        z_coords = sub_physical_pos[:, 2].unsqueeze(1)
        node_features = torch.cat([node_features, z_coords], dim=1)

        if self.fully_connected:
            node_pos_abs = template['relative_pos'] + torch.tensor([0.0, float(iy), float(ix)])
            data = Data(x=node_features, pos=node_pos_abs, y=node_targets)
            graph_data = self.radius_transform(data)
        else:
            edge_index = template['edge_index']
            if edge_index.numel() > 0:
                row, col = edge_index
                edge_vectors = sub_physical_pos[row] - sub_physical_pos[col]
                edge_attr = edge_vectors.norm(dim=1).unsqueeze(1)
            else:
                edge_attr = torch.empty((0, 1), dtype=torch.float32)

            graph_data = Data(
                x=node_features, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                y=node_targets
            )

        graph_data.central_mask = template['central_mask']
        graph_data.u = self.static_u

        return graph_data