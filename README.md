# GraphNet-3D-NLTE: Fast 3D Non-LTE Populations with Graph Neural Networks

**GraphNet-3D-NLTE** is a deep learning framework that accelerates the computation of non-LTE (non-local thermodynamic equilibrium) atomic level populations in 3D solar and stellar atmospheres.

This is the code of [Vicente Arévalo, Asensio Ramos & Díaz Baso (2026), *Accelerating 3D Non-LTE Synthesis with Graph Neural Networks*, A&A 710, A256](https://doi.org/10.1051/0004-6361/202660077) ([arXiv:2605.09543](https://arxiv.org/abs/2605.09543)), which extends the 1.5D method of [Vicente Arévalo et al. (2022)](https://doi.org/10.3847/1538-4357/ac53b3) to **3D geometries**. A graph neural network (GNN) predicts the non-LTE populations of a Ca II model atom, as the log10 fractions of its six levels (Ca II 4s, 3d3/2, 3d5/2, 4p1/2, 4p3/2 and Ca III), without solving the 3D radiative transfer problem. The results in the paper were obtained with an earlier version of this code; this branch adds the changes described below and in release [3D_GNN-v1.0](https://github.com/andreuva/graphnet_nlte/releases/tag/3D_GNN-v1.0).

## Table of Contents

* [Background & Motivation](#background--motivation)
* [Architecture: From 1D to 3D](#architecture-from-1d-to-3d)
* [Key Features](#key-features)
* [Installation](#installation)
* [Dataset Preparation](#dataset-preparation)
* [Usage](#usage)
* [Citation](#citation)
* [License](#license)
* [Future Work](#future-work)

## Background & Motivation

The correct inference of spectral lines formed in the chromosphere and transition region requires solving the non-LTE problem. This is computationally expensive because the problem is **non-linear** (the radiation field depends on the populations and vice versa) and **non-local** (radiation couples distant points of the atmosphere).

Standard methods like MALI (Multilevel Accelerated Lambda Iteration) are robust but slow, which makes them prohibitively expensive in 3D, in inversions and in time-series analysis.

Vicente Arévalo et al. (2022) proposed a graph network that predicts the atomic level populations without solving the non-LTE problem, and applied it to 1.5D (column-by-column) atmospheres, which neglect horizontal radiative transfer. Vicente Arévalo et al. (2026) extend it to the full 3D domain: the atmosphere is discretized as a directed graph whose nodes encode the physical properties and whose edges encode the spatial distances, so that the fast synthesis takes 3D radiative transfer effects into account.

## Architecture: From 1D to 3D

The core architecture follows the "Encode-Process-Decode" paradigm of Battaglia et al. (2018), as applied to solar physics by Vicente Arévalo et al. (2022, 2026).

### 1.5D Implementation (Vicente Arévalo et al. 2022)

* **Nodes:** Physical properties at a specific depth.

* **Edges:** Connected vertical neighbors in the optical depth scale.

* **Limitation:** Ignores horizontal interactions.

### 3D Implementation (Vicente Arévalo et al. 2026, This Code)

* **Nodes:** The 49 columns of a disc of radius 4 hops inside the 9 x 9 stencil around the predicted column (`x_range_graph`, `y_range_graph`, `stencil_radius`), at the levels of a vertical grid. A hop is 1 to `max_stride` pixels (47.6 km each on Bifrost). Inputs per node (12): vx, vy, vz divided by 10 km/s; log10 T, log10 rho and log10 P standardized with one mean and standard deviation of the training cube and clipped at 5 sigma; and the six log10 LTE (Saha-Boltzmann) fractions of the Ca levels computed with Lightweaver from the atmosphere's own T and n_e. No magnetic field and no height coordinate. The normalization statistics are fitted once on the training cube and stored in the checkpoint.

* **Edges:** Defined by physical distance, in a star topology around the central (predicted) column. A node of a neighbour column and a node of the central column are linked, in both directions, when their distance is at most `edge_radius_Mm` (1 Mm); two nodes of the central column are linked when their heights differ by at most `central_reach_Mm` (1.5 Mm) or their levels are adjacent. Neighbour columns are not linked to each other, and there are no self loops. Edge features: the vertical distance, the horizontal distance and |dr|, in Mm. Because the reaches are in Mm, the receptive field is the same for every hop and vertical grid.

* **Message Passing:** `n_message_passing_steps` (8) steps of edge and node updates with a latent size of 128 (`graphnet.py`). Information propagates along the central column and between the central column and its neighbours, which mimics 3D radiative transfer effects. Only the nodes of the central column are predicted and enter the loss; the neighbour columns provide context.

* **Outputs and loss:** log10 of the six level fractions (log10 of a softmax, so the fractions sum to one); the absolute populations are the fractions times the total Ca density (9.65e14 x rho [kg m^-3] cm^-3, the abundance of the training targets). The loss is the mean squared error of the log10 fractions, each node weighted by min(1, f_CaII / 0.01) but never less than 0.1 (`relevance_threshold`, `coronal_weight` in `conf.dat`), so that the corona (no Ca II) does not dominate.

* **Vertical grids** (`vertical_grids.py`, heights above tau500 = 1, the same for Bifrost and MURaM): the levels are spent where the Ca II lines form, and only a few are left in the corona; the bottom and top native levels are always kept. Every training sample draws a random two-zone grid (60-110 km spacing up to a transition at 3.8-4.8 Mm, 5-10 coronal levels, jitter and level removal), random hops of 1 to `max_stride` (5) pixels per axis and a rotation of the volume and of (vx, vy) by a random multiple of 90 degrees about the vertical axis (the only augmentation). Validation uses the fixed grid `grid` of `conf.dat`: `dense:125` = 125 km steps from the bottom level up to 4.5 Mm, then 7 coronal levels (49 levels on Bifrost). Every node takes its inputs by linear interpolation in height between the two native levels around it.

* **Training recipe** (values in `conf.dat`): AdamW (learning rate 1e-3, weight decay 0.01), one epoch of linear warmup followed by a cosine decay per batch, gradient clipping at 1, an exponential moving average (EMA) of the weights validated next to the raw weights, and a fresh random 15 % of the training columns every epoch. Validation measures the error in dex on the held-out corner of the snapshot (beyond 70 % of the snapshot in x and in y), on the columns at least 32 px away from the training region.

## Key Features

* **3D Spatial Awareness:** unlike 1D column-by-column approaches, this code treats the atmosphere as a 3D connected graph, accounting for horizontal radiative transfer effects indirectly through learned spatial correlations.

* **Arbitrary Grids:** the graph is built from physical distances, so the same network runs on any vertical grid and at any horizontal resolution (e.g. MURaM at its native resolution of 23.4 km).

* **Differentiable:** the network is fully differentiable, so it can be integrated into inversion codes to compute response functions by backpropagation.

* **Speed:** about 20 ms per column on one GPU (measured on the `dense:60` grid, in batches of 48 columns).

## Installation

Clone the repository and install the dependencies. A GPU is strongly recommended for inference and required in practice for training.

```bash
git clone https://github.com/andreuva/graphnet_nlte.git
cd graphnet_nlte
git checkout 3D_GNN
conda env create -f environment.yml        # environment 'gph' (Python 3.12)
conda activate gph
```

**Key Dependencies:**

* `torch` and `torch_geometric` (the graph network)
* `lightweaver` (the LTE fractions of the inputs and the spectral syntheses; pinned to a commit in `environment.yml`)
* `conda` (environment management)
* `cuda` (GPU acceleration)

## Dataset Preparation

The model requires training data consisting of model atmospheres and their "ground truth" populations from a 3D non-LTE radiative transfer code (e.g. Multi3D, PORTA or another).

1. **Input:** 3D MHD snapshots (here Bifrost `en024048_hion`, snapshot 385: 504 x 504 columns of 47.6 km, 24 Mm, and the 425 native levels 52-476 of the FITS cubes, from -0.58 to 12.5 Mm above tau500 = 1).

2. **Target:** the 3D non-LTE populations of the snapshot computed with Multi3D (Vicente Arévalo et al. 2026), in `AR_385_CaII_5L_pops.dat`: the five Ca II levels and Ca III in cm^-3.

**Data Structure:**
The code expects the data in the format below; other simulations can be adapted. Memory-mapped float32 cubes (z, y, x, components) in `../data_train_v2/`: `AR_385_temp.dat`, `_veloc.dat`, `_ne.dat`, `_mass.dat`, `_press.dat`, `_nh.dat`, `_np.dat` and `_B.dat` in SI units, the targets, and a `MANIFEST.json`; the heights of the native levels are in `../en024048_hion/grid_bifrost.npz`. The earlier `data_train` cubes were written with `bifrost_raw_read_test.ipynb`, which read Bz from the By file (fixed now); `data_train_v2` was regenerated from the FITS files in `en024048_hion/atmos/` on branch `new_topologies` (`dataset_scripts/make_data_train.py`) and differs from `data_train` only in Bz.

* **Nodes:** vx, vy, vz, T, rho, P and the LTE fractions from T and n_e, normalized in `normalization.py`.

* **Edges:** the connectivity and the edge features (vertical, horizontal and total distance in Mm), built in `Dataset.py`.

## Usage

### 1. Training

Set the parameters in `conf.dat` (each key is commented there) and run:

```bash
CUDA_VISIBLE_DEVICES=<free gpu> python gnn_train.py conf.dat [seed]     # -> checkpoints/v3_lte_seed<seed>/
```

The training minimizes the weighted mean squared error of the log10 level fractions described above. The `*_best.pth` checkpoint (the epoch with the lowest held-out error, of the raw or the EMA weights) holds the weights, the configuration, the input list and the normalization statistics. Checkpoints written by earlier versions of the code do not load.

### 2. Inference (Prediction)

Predict the populations on the held-out corner of the Bifrost snapshot (or on all of it with `full`), and on a MURaM snapshot at its native resolution. Run the scripts from the repository folder. `gnn_inference.py` and `muram_inference.py` write a folder next to the checkpoint; the plotting and comparison scripts write into the folder they are given; `compare_datasets.py` and `compare_height_stratification.py` write their figures into the current folder:

```bash
python gnn_inference.py <checkpoint> [stride] [test|full] [device] [grid]                    # -> bifrost_stride{s}_{split}_{grid}/
python plot_scripts/inference_plots.py <bifrost folder>
python synthesis.py <bifrost folder>; python plot_scripts/synthesis_plots.py <bifrost folder>   # Ca II 8542
python muram_comparison/muram_inference.py <checkpoint> [stride] [full | y0,y1,x0,x1] [device] [grid]   # -> muram_stride{s}_{grid}_{crop}/
python muram_comparison/compare_populations.py <muram folder>                                 # against the 1.5D NLTE reference
python muram_comparison/muram_synthesis.py <muram folder> [patch size]                        # default patch 32
python muram_comparison/compare_height_stratification.py [<muram folder>]
python muram_comparison/compare_datasets.py <checkpoint> [grid]                               # Bifrost vs MURaM grids and inputs
```

Defaults: the CPU; on Bifrost the held-out corner and the training hop (`max_stride`, 5 px = 238 km); on MURaM the full snapshot and the same hop in km (10 px). The grid defaults to `dense:60` in every script, whatever the training grid: pass it explicitly to evaluate on the training grid, e.g.

```bash
python gnn_inference.py checkpoints/v3_lte_seed777/<date>_best.pth 5 test cuda dense:125
```

The notebooks (`gnn_inference.ipynb`, `synthesis.ipynb`, ...) use the code of commit fbaba82, and `plot_scripts/plot_graph.py` illustrates the index-space graph of that code.

## Citation

If you use this code in your research, please cite the paper describing the 3D method:

```bibtex
@article{Vicente_Arevalo_2026,
	doi = {10.1051/0004-6361/202660077},
	url = {https://doi.org/10.1051/0004-6361/202660077},
	year = 2026,
	publisher = {EDP Sciences},
	volume = {710},
	pages = {A256},
	author = {A. Vicente Arévalo and A. Asensio Ramos and C. J. Díaz Baso},
	title = {Accelerating 3D Non-LTE Synthesis with Graph Neural Networks},
	journal = {Astronomy \& Astrophysics},
	eprint = {2605.09543},
	archivePrefix = {arXiv},
	primaryClass = {astro-ph.SR}
}
```

and the paper that introduced the graph network for non-LTE populations (1.5D):

```bibtex
@article{Vicente_Arevalo_2022,
	doi = {10.3847/1538-4357/ac53b3},
	url = {https://doi.org/10.3847/1538-4357/ac53b3},
	year = 2022,
	month = {mar},
	publisher = {American Astronomical Society},
	volume = {928},
	number = {2},
	pages = {101},
	author = {A. Vicente Arévalo and A. Asensio Ramos and S. Esteban Pozuelo},
	title = {Accelerating Non-LTE Synthesis and Inversions with Graph Networks},
	journal = {The Astrophysical Journal}
}
```

Please also cite this repository if you use the 3D implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Work

### Integration with Inversion Codes

The network is meant to act as an "oracle" for inversion codes (like Hazel2 or STiC): instead of solving the statistical equilibrium equations iteratively, the inversion code queries the network. This is not implemented yet; the pseudocode below shows the intended use (`graphnet3d.GNModel` does not exist yet):

```python
import torch
from graphnet3d import GNModel

# Load trained model
model = GNModel.load('weights/ca_ii_best.pt')

# Inversion loop
for iteration in range(max_iter):
    # Predict the level populations
    pops = model(current_atmosphere)

    # Synthesize spectra using the populations (e.g., with Lightweaver or SIR)
    spectra = synthesize(current_atmosphere, pops)

    # ... compute loss and update atmosphere ...
```
