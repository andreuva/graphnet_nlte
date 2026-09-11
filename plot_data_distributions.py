import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset


def parse_conf(conf_path):
    params = {}
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    k_str = k.strip()
                    v_str = v.strip()
                    try:
                        params[k_str] = int(v_str)
                    except ValueError:
                        try:
                            params[k_str] = float(v_str)
                        except ValueError:
                            params[k_str] = v_str
    return params


def main():
    parser = argparse.ArgumentParser(description="Plot dataset feature and target distributions directly from Dataset module.")
    parser.add_argument("--datadir", default="/dat/andreuva/gpu/graphnet/data_1d_si/", help="Directory containing dataset .pkl files")
    parser.add_argument("--prefix", default="test", help="Dataset prefix (e.g., train, test, validation)")
    parser.add_argument("--conf", default="conf.dat", help="Configuration file (e.g. conf.dat)")
    parser.add_argument("--nsamples", type=int, default=5000, help="Number of atmosphere models to sample for fast plotting (0 or negative for all)")
    parser.add_argument("--output", default="data_distributions.png", help="Output plot image filename")
    args = parser.parse_args()

    # Load hyperparameters from conf file if available
    hyperparameters = parse_conf(args.conf)
    if hyperparameters:
        print(f"Loaded hyperparameters from {args.conf}: {hyperparameters}")
    else:
        hyperparameters = {'node_input_size': 5, 'edge_input_size': 1}
        print(f"Config file {args.conf} not found/empty. Using default hyperparameters: {hyperparameters}")

    print(f"Instantiating Dataset from {args.datadir} (prefix: {args.prefix})...")
    dataset = Dataset(hyperparameters, datadir=args.datadir, prefix=args.prefix)

    n_models = len(dataset.nodes)
    if args.nsamples > 0 and args.nsamples < n_models:
        sample_indices = range(args.nsamples)
    else:
        sample_indices = range(n_models)

    print(f"Extracting normalized features from {len(sample_indices)} models...")

    # Concatenate nodes, edges, and targets across sampled models
    sampled_nodes = [np.asarray(dataset.nodes[i]) for i in sample_indices if dataset.nodes[i] is not None]
    sampled_edges = [np.asarray(dataset.edges[i]) for i in sample_indices if dataset.edges[i] is not None]
    sampled_targets = [np.asarray(dataset.target[i]) for i in sample_indices if dataset.target[i] is not None]

    nodes_arr = np.concatenate(sampled_nodes, axis=0)  # Shape: (Total_Nodes, node_input_size)
    edges_arr = np.concatenate(sampled_edges, axis=0)  # Shape: (Total_Edges, edge_input_size)
    targets_arr = np.concatenate(sampled_targets, axis=0).ravel()  # Flattened targets

    # Define plot layout for node features, edge features, and target
    plot_dict = {}

    # Node feature titles
    node_feature_names = [
        "Node 0: Norm log10(T)",
        "Node 1: Norm z" if dataset.z_activated else "Node 1: Norm log10(tau)",
        "Node 2: Norm log10(ne)",
        "Node 3: Norm vturb",
        "Node 4: Norm vlos",
    ]

    for f_idx in range(nodes_arr.shape[1]):
        name = node_feature_names[f_idx] if f_idx < len(node_feature_names) else f"Node Feature {f_idx}"
        plot_dict[name] = nodes_arr[:, f_idx]

    # Edge feature titles
    for e_idx in range(edges_arr.shape[1]):
        if e_idx == 0:
            name = "Edge 0: Norm Delta z" if dataset.z_activated else "Edge 0: Norm Delta tau"
        else:
            name = f"Edge {e_idx}: Norm Delta tau"
        plot_dict[name] = edges_arr[:, e_idx]

    # Target
    plot_dict["Target: Norm log departure (y)"] = targets_arr

    # Print summary statistics
    print("\n--- Feature Statistics (as processed by Dataset) ---")
    for name, data in plot_dict.items():
        data_clean = data[np.isfinite(data)]
        print(f"{name:32s} -> min={np.min(data_clean):8.3f}, max={np.max(data_clean):8.3f}, mean={np.mean(data_clean):8.3f}, std={np.std(data_clean):8.3f}")

    # Plotting
    n_plots = len(plot_dict)
    cols = 4
    rows = (n_plots + cols - 1) // cols

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, (name, val) in enumerate(plot_dict.items()):
        ax = axes[i]
        c = colors[i % len(colors)]
        val_clean = val[np.isfinite(val)]
        ax.hist(val_clean, bins=80, color=c, alpha=0.7, edgecolor='black', linewidth=0.5, density=True)

        mean_val, std_val = np.mean(val_clean), np.std(val_clean)
        min_val, max_val = np.min(val_clean), np.max(val_clean)

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        info_str = f"Mean: {mean_val:6.2f}\nStd:  {std_val:6.2f}\nMin:  {min_val:6.2f}\nMax:  {max_val:6.2f}"
        ax.text(0.95, 0.95, info_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Hide unused subplots
    for j in range(len(plot_dict), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\nPlot successfully saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
