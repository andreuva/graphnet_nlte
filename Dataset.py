# DATASET CLASS DEFINITION

import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.transforms import RadiusGraph


class Dataset(torch.utils.data.Dataset):

    def __init__(self, list_X: list, list_Y: list, radius_neibourghs=1.5, device='cpu'):
        """
        Dataset for the depth stratification
        """
        super(Dataset, self).__init__()

        np.random.seed(777)
        self.device = device

        # check that all the cubes in the list have the same (nx, ny, nz, *) dimensions
        if len(list_X) < 1:
            raise ValueError("You should provide some datacube")
        
        self.shape_cube = np.shape(list_X[0])[:-1]
        self.nz, self.ny, self.nx = self.shape_cube

        values = []
        print("  Joining the cubes into a data structure")
        for ind_cube, cube in enumerate(list_X):
            if np.shape(cube)[:-1] != self.shape_cube:
                raise ValueError(f"Dimensions of cube {ind_cube} do not match")

            values.append(cube.reshape(self.nz*self.ny*self.nx, -1))
        
        targets = []
        for ind_cube, cube in enumerate(list_Y):
            if np.shape(cube)[:-1] != self.shape_cube:
                raise ValueError(f"Dimensions of target cube {ind_cube} do not match")

            targets.append(cube.reshape(self.nz*self.ny*self.nx, -1))

        self.values_graph = torch.tensor(np.hstack(values))
        self.targets_graph = torch.tensor(np.hstack(targets))

        print("  Creating the grid to calculate the conections of the graph")
        self.xx = np.linspace(0, self.nx - 1, self.nx)
        self.yy = np.linspace(0, self.ny - 1, self.ny)
        self.zz = np.linspace(0, self.nz - 1, self.nz)
        zv, yv, xv = np.meshgrid(self.zz, self.yy, self.xx, indexing='ij')
        self.grid = np.stack([zv.ravel(), yv.ravel(), xv.ravel()], axis=1)
        self.grid = torch.tensor(self.grid, dtype=torch.float)

        self.u = torch.zeros((1,1), dtype=torch.float)

        print("  Storing radius for manual graph construction")
        self.radius_neighbors = radius_neibourghs
        print("...")
        # Create initial graph structure - edges will be computed per subgraph
        self.full_data = torch_geometric.data.Data(x=self.values_graph,
                                                   pos=self.grid,
                                                   y=self.targets_graph)

    def create_sample_indices(self, xpos, ypos, xdim=1, ydim=1):
        # Calculate the valid x and y ranges for the subgrid
        x_start = max(0, xpos - xdim)
        x_end = min(self.nx, xpos + xdim + 1)
        y_start = max(0, ypos - ydim)
        y_end = min(self.ny, ypos + ydim + 1)

        subgrid_indices = [
            k * self.ny * self.nx + j * self.nx + i
            for k in range(self.nz)
            for j in range(y_start, y_end)
            for i in range(x_start, x_end)
        ]

        return np.array(subgrid_indices)

    def grid_to_graph_manual(self, grid_points, values=None, targets=None, r=1.5, xpos=None, ypos=None):
        """
        Converts a set of 3D grid points to a PyTorch Geometric graph
        by manually calculating edges between nodes closer than a given radius.
        """
        if values is None:
            values = grid_points
        if targets is None:
            targets = grid_points

        # --- MANUAL EDGE CONSTRUCTION ---
        # 1. Calculate pairwise distances between all points
        dist_matrix = torch.cdist(grid_points, grid_points)

        # 2. Find pairs (i, j) where 0 < distance <= r
        # We exclude 0 to avoid self-loops.
        edge_indices_tuple = torch.where((dist_matrix > 0) & (dist_matrix <= r))

        # 3. Stack the indices to create the edge_index tensor of shape [2, num_edges]
        edge_index = torch.stack(edge_indices_tuple, dim=0)
        # --- END OF MANUAL CONSTRUCTION ---

        # Create a Data object with the manually computed edges
        graph_data = Data(x=values, pos=grid_points, y=targets, edge_index=edge_index)

        # Find and filter for central nodes (using the corrected logic from the previous answer)
        if xpos is not None and ypos is not None:
            central_nodes = torch.where(
                (grid_points[:, 2] == xpos) & (grid_points[:, 1] == ypos)
            )[0]

            # Filter edges to keep only those connected to central nodes
            if len(central_nodes) > 0:
                edge_mask = torch.from_numpy(
                    np.isin(graph_data.edge_index[0, :], central_nodes) |
                    np.isin(graph_data.edge_index[1, :], central_nodes)
                )
                graph_data.edge_index = graph_data.edge_index[:, edge_mask]

        return graph_data

    def __len__(self):
        return self.nx*self.ny

    def __call__(self):
        ix = np.random.randint(1, self.nx + 1)
        iy = np.random.randint(1, self.ny + 1)
        indices = self.create_sample_indices(ix, iy)

        # Get subgrid positions, features, and targets
        sub_pos = self.grid[indices]
        sub_features = self.values_graph[indices]
        sub_targets = self.targets_graph[indices]

        # Use manual graph construction
        subgraph = self.grid_to_graph_manual(sub_pos, sub_features, sub_targets, r=self.radius_neighbors, xpos=ix, ypos=iy)
        subgraph.u = self.u
        return subgraph

    def __getitem__(self, index):
        np.random.seed(index)
        ix = np.random.randint(1, self.nx + 1)
        iy = np.random.randint(1, self.ny + 1)
        indices = self.create_sample_indices(ix, iy)

        # Get subgrid positions, features, and targets
        sub_pos = self.grid[indices]
        sub_features = self.values_graph[indices]
        sub_targets = self.targets_graph[indices]

        # Use manual graph construction
        subgraph = self.grid_to_graph_manual(sub_pos, sub_features, sub_targets, r=self.radius_neighbors, xpos=ix, ypos=iy)
        subgraph.u = self.u
        return subgraph


class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, list_X: list, list_Y: list, radius_neighbors=1.5, xdim=1, ydim=1, pos_file=None, seed=777, train_ratio=0.9, split='train', device='cpu'):
        super(EfficientDataset, self).__init__()
        self.device = device
        self.radius = radius_neighbors
        self.xdim = xdim
        self.ydim = ydim

        # Store data as numpy arrays to keep them on CPU until needed
        self.features = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_X], axis=1)
        self.targets = np.concatenate([arr.reshape(-1, arr.shape[-1]) for arr in list_Y], axis=1)

        self.nz, self.ny, self.nx = list_X[0].shape[:-1]

        if pos_file is None:
            self.xx, self.yy, self.zz = np.linspace(0, self.nx - 1, self.nx), np.linspace(0, self.ny - 1, self.ny), np.linspace(0, self.nz - 1, self.nz)
        else:
            grid_pos = np.load(pos_file)
            self.xx, self.yy, self.zz = grid_pos['x']*10, grid_pos['y']*10, grid_pos['z']*10
            self.zz = np.interp(np.linspace(0, len(self.zz), self.nz), np.arange(len(self.zz)), self.zz)
        xgrid, ygrid, zgrid = np.meshgrid(self.zz, self.yy, self.xx, indexing='ij')
        self.grid_pos = torch.tensor(np.stack([xgrid.ravel(), ygrid.ravel(), zgrid.ravel()], axis=1), dtype=torch.float)
        
        # Store radius for manual graph construction
        # Note: RadiusGraph transform replaced with manual construction

        valid_ix = np.arange(xdim, self.nx - xdim)
        valid_iy = np.arange(ydim, self.ny - ydim)
        all_indices = [(x, y) for x in valid_ix for y in valid_iy]

        # Shuffle and split
        np.random.seed(seed)
        np.random.shuffle(all_indices)

        split_idx = int(train_ratio * len(all_indices))
        if split == 'train':
            self.sample_centers = all_indices[:split_idx]
        elif split == 'test':
            self.sample_centers = all_indices[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'test'")

        print(f'{split.capitalize()} dataset created. Total samples: {len(self.sample_centers)}')
        print(f'Features shape: {self.features.shape}, Targets shape: {self.targets.shape}')

    def __len__(self):
        return len(self.sample_centers)//10

    def grid_to_graph_manual(self, grid_points, values=None, targets=None, r=1.5, xpos=None, ypos=None):
        """
        Converts a set of 3D grid points to a PyTorch Geometric graph
        by manually calculating edges between nodes closer than a given radius.
        """
        if values is None:
            values = grid_points
        if targets is None:
            targets = grid_points

        # --- MANUAL EDGE CONSTRUCTION ---
        # 1. Calculate pairwise distances between all points
        dist_matrix = torch.cdist(grid_points, grid_points)

        # 2. Find pairs (i, j) where 0 < distance <= r
        # We exclude 0 to avoid self-loops.
        edge_indices_tuple = torch.where((dist_matrix > 0) & (dist_matrix <= r))

        # 3. Stack the indices to create the edge_index tensor of shape [2, num_edges]
        edge_index = torch.stack(edge_indices_tuple, dim=0)
        # --- END OF MANUAL CONSTRUCTION ---

        # Create a Data object with the manually computed edges
        graph_data = Data(x=values, pos=grid_points, y=targets, edge_index=edge_index)

        # Find and filter for central nodes (using the corrected logic from the previous answer)
        if xpos is not None and ypos is not None:
            central_nodes = torch.where(
                (grid_points[:, 2] == xpos) & (grid_points[:, 1] == ypos)
            )[0]

            # Filter edges to keep only those connected to central nodes
            if len(central_nodes) > 0:
                edge_mask = torch.from_numpy(
                    np.isin(graph_data.edge_index[0, :], central_nodes) |
                    np.isin(graph_data.edge_index[1, :], central_nodes)
                )
                graph_data.edge_index = graph_data.edge_index[:, edge_mask]

        return graph_data

    def __getitem__(self, index):
        ix, iy = self.sample_centers[index]

        y_range = np.arange(iy - self.ydim, iy + self.ydim + 1)
        x_range = np.arange(ix - self.xdim, ix + self.xdim + 1)
        k_range = np.arange(self.nz)

        # Create a grid of coordinates for the sub-volume
        kv, yv, xv = np.meshgrid(k_range, y_range, x_range, indexing='ij')

        # Flatten and stack to get node positions
        node_pos = torch.tensor(np.stack([kv.ravel(), yv.ravel(), xv.ravel()], axis=1), dtype=torch.float)

        # Calculate flat indices from the coordinate grid to slice the original numpy arrays
        flat_indices = (kv.ravel() * self.ny * self.nx + yv.ravel() * self.nx + xv.ravel())

        # 2. Get the features and targets for ONLY this sub-grid
        node_features = torch.tensor(self.features[flat_indices], dtype=torch.float)
        node_targets = torch.tensor(self.targets[flat_indices], dtype=torch.float)

        # 3. Use manual graph construction instead of RadiusGraph transform
        graph_data = self.grid_to_graph_manual(node_pos, node_features, node_targets, r=self.radius, xpos=ix, ypos=iy)

        # Add proper edge attributes if they don't exist
        if graph_data.edge_attr is None:
            row, col = graph_data.edge_index
            edge_vectors = graph_data.pos[row] - graph_data.pos[col]
            graph_data.edge_attr = edge_vectors.norm(dim=1).unsqueeze(1)

        # Add a place holder for the global attributes 'u' because the model needs it
        graph_data.u = torch.zeros((1, 1), dtype=torch.float)
        return graph_data