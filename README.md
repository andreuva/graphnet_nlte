# GraphNet-3D-NLTE: Fast 3D Non-LTE Populations with Graph Neural Networks

**GraphNet-3D-NLTE** is a deep learning framework designed to accelerate the computation of non-LTE (Local Thermodynamic Equilibrium) atomic populations in 3D solar and stellar atmospheres.

Building upon the 1D architecture proposed by [Vicente Arévalo et al. (2022)](https://doi.org/10.3847/1538-4357/ac53b3), this project extends the concept to fully **3D geometries**. It utilizes Graph Neural Networks (GNNs) to predict the non-LTE level populations of a Ca II model atom (the log10 fractions of its six levels: 4s, 3d3/2, 3d5/2, 4p1/2, 4p3/2 and Ca III) orders of magnitude faster than traditional 3D radiative transfer solvers.

## 📖 Table of Contents

* [Background & Motivation](#-background--motivation)
* [Key Features](#-key-features)
* [Installation](#️-installation)
* [Dataset Preparation](#-dataset-preparation)
* [Usage](#-usage)
* [Architecture (From 1D to 3D)](#-architecture-from-1d-to-3d)
* [Citation](#-citation)

## 🔭 Background & Motivation

The correct inference of spectral lines formed in the chromosphere and transition region requires solving the non-LTE problem. This is traditionally computationally expensive because the problem is **non-linear** (properties depend on populations) and **non-local** (radiation couples distant points).

Standard methods like MALI (Multilevel Accelerated Lambda Iteration) are robust but slow, limiting their use and making them prohibitively expensive in 3D inversions or time-series analysis.

This project implements the solution proposed by Vicente Arévalo et al. (2022):

> "We propose to build and train a graph network that quickly predicts the atomic level populations without solving the 3D non-LTE problem, and therefore having acces to fast and accurate synthesis having into account 3D RT effects." 
> 
> 

While the original paper focused on 1D plane-parallel atmospheres, this codebase realizes the "future work" mentioned in the manuscript: applying the method to higher dimensions and arbitrary topologies.

## 🧠 Architecture: From 1D to 3D

The core architecture follows the "Encode-Process-Decode" paradigm described in Battaglia et al. (2018) and applied to solar physics by Vicente Arévalo et al. (2022) .

### 1D Implementation (Paper)

* **Nodes:** Physical properties at a specific depth.

* **Edges:** Connected vertical neighbors in the optical depth scale.

* **Limitation:** Ignores horizontal interactions.

### 3D Implementation (This Code)

* **Nodes:** Same physical properties, but representing a voxel: the 49 columns of a disc of radius 4 inside the 9x9 stencil around the predicted column, with hops of `max_stride` pixels, at the levels of a vertical grid. Inputs per node (12): vx, vy, vz in units of 10 km/s; log10 T, log10 rho and log10 P standardized with one mean and standard deviation of the training cube; and the six log10 LTE (Saha-Boltzmann) fractions of the Ca levels computed from the atmosphere's own T and n_e. No magnetic field and no height coordinate.

* **Edges:** Connections are defined by physical distance (star topology): a node is linked to the nodes of the central column within a sphere of radius 1 Mm around them (|dr| <= `edge_radius_Mm`), and two nodes of the central column within 2 Mm in height (`central_reach_Mm`, adjacent levels always). Neighbour columns are not linked to each other. Edge features: the vertical and horizontal distances and |dr| in Mm.

* **Message Passing:** Information propagates primarily vertically (due to stratification) but also horizontally, mimicking 3D radiative transfer effects, and only to the center column, neibourgs are only used for context and not actually predicted.

* **Outputs and loss:** log10 of the six level fractions (log10 of a softmax, so the fractions sum to one); the absolute populations are the fractions times the total Ca density (9.65e14 x rho [kg m^-3] cm^-3, the abundance of the training targets). The loss is the mean squared error of the log10 fractions, each node weighted by min(1, f_CaII / 0.01) but never less than 0.1 (`relevance_threshold`, `coronal_weight` in `conf.dat`), so that the corona (no Ca II) does not dominate.

* **Vertical grids** (`vertical_grids.py`, heights above tau500 = 1, the same for Bifrost and MURaM): the levels are spent where the Ca II lines form and only a few are left in the corona. Every training sample draws a random two-zone grid (60-110 km spacing up to a transition at 3.8-4.8 Mm, 5-10 coronal levels, jitter and level removal; bottom and top levels kept), random hops of 1-4 pixels per axis and a rotation of the volume about the vertical axis (the only augmentation). Validation and inference use the fixed `dense:60` grid (60 km steps up to 4.5 Mm and 7 coronal levels).

* **Training recipe:** AdamW, one epoch of linear warmup and cosine decay, gradient clipping, an exponential moving average of the weights validated next to the raw weights, a fresh random subset of the training columns every epoch, and validation in dex on a held-out corner of the snapshot.

## ✨ Key Features

* **3D Spatial Awareness:** unlike 1D column-by-column approaches, this code treats the atmosphere as a 3D connected graph, accounting for horizontal radiative transfer effects indirectly through learned spatial correlations.

* **Arbitrary Grids:** The Graph Network architecture allows for variable grids. Nodes can be separated different horizontal distances, better generalizing to different resolution datasets (e.g. MURaM at its native resolution).

* **Differentiable:** The network is fully differentiable, allowing seamless integration into inversion codes for calculation of Response Functions via backpropagation.

* **Speed:** Predicts the populations of a column in about 20 ms on one GPU (`dense:60` grid, batches of 48 columns).

## ⚙️ Installation

Clone the repository and install the dependencies. A GPU is strongly recommended for inference and unavoidable for training.

```bash
git clone https://github.com/andreuva/graphnet_nlte.git
cd graphnet_nlte
git checkout new_topologies_simple
conda env create -f environment.yml        # environment 'gph'
conda activate gph
```

**Key Dependencies:**

* `torch` & `torch_geometric` (for Graph Network implementation )
* `lightweaver` (the LTE fractions of the inputs and the spectral syntheses)
* `conda` for environment management
* `cuda` (if using GPU acceleration)

## 💾 Dataset Preparation

The model requires training data consisting of model atmospheres and their corresponding "ground truth" populations computed by a specific non-LTE code (e.g., Multi3D, PORTA, or other).

1. **Input:** 3D MHD snapshots (e.g., Bifrost `en024048_hion`, snapshot 385: 504 x 504 columns of 47.6 km, the 425 native levels 52-476 of the FITS cubes).

2. **Target:** Populations calculated via a trusted radiative transfer code (`AR_385_CaII_5L_pops.dat`, the five Ca II levels and Ca III in cm^-3, a 3D non-LTE solution of the snapshot whose code and setup are not documented).

**Data Structure:**
The code expects data in BIFROST format, but easy adaptation to other formats is possible: memory-mapped float32 cubes (z, y, x, components) in `../data_train_v2/` (`AR_385_temp.dat`, `_veloc.dat`, `_ne.dat`, `_mass.dat`, `_press.dat`, `_nh.dat`, `_np.dat`, `_B.dat`, in SI units, and the targets), and the heights of the native levels in `../en024048_hion/grid_bifrost.npz`. The earlier `data_train` cubes were written with `bifrost_raw_read_test.ipynb`, which read Bz from the By file (fixed now); `data_train_v2` was regenerated from the FITS files in `en024048_hion/atmos/` on branch `new_topologies` (`dataset_scripts/make_data_train.py`, see its `MANIFEST.json`) and differs from `data_train` only in Bz.

* **Nodes:** Features vx, vy, vz, T, rho, P and the LTE fractions from T and n_e (normalized in `normalization.py`)
 
* **Edges:** Connectivity indices and edge features (vertical, horizontal and total distance in Mm), built in `Dataset.py`.


## 🚀 Usage

### 1. Training

To train a new 3D model (e.g., for Calcium II), configure the parameters in `conf.dat` (self explanatory fields) and run:

```bash
CUDA_VISIBLE_DEVICES=<free gpu> python gnn_train.py conf.dat [seed]     # -> checkpoints/v3_lte_seed<seed>/

```

The training process optimizes the Mean Squared Error (MSE) between the predicted and computed log10 level fractions, weighted by the Ca II relevance of each node. The `*_best.pth` checkpoint (the epoch with the lowest held-out error, raw or averaged weights) holds the weights, the configuration, the input list and the normalization statistics.

### 2. Inference (Prediction)

Generate populations on the held-out corner of the Bifrost snapshot (or on all of it with `full`), and on a MURaM snapshot at its native resolution. Run the scripts from the repository folder: `gnn_inference.py` and `muram_inference.py` write a folder next to the checkpoint, the plotting and comparison scripts write into the folder they are given, and `compare_datasets.py` and `compare_height_stratification.py` write their figures into the current folder:

```bash
python gnn_inference.py <checkpoint> [stride] [test|full] [device] [grid]                    # -> bifrost_stride{s}_{split}_{grid}/
python plot_scripts/inference_plots.py <bifrost folder>
python synthesis.py <bifrost folder>; python plot_scripts/synthesis_plots.py <bifrost folder>   # Ca II 8542
python muram_comparison/muram_inference.py <checkpoint> [stride] [full | y0,y1,x0,x1] [device] [grid]   # -> muram_stride{s}_{grid}_{crop}/
python muram_comparison/compare_populations.py <muram folder>                                 # against the 1.5D reference
python muram_comparison/muram_synthesis.py <muram folder> [patch size]
python muram_comparison/compare_height_stratification.py [<muram folder>]
python muram_comparison/compare_datasets.py <checkpoint> [grid]                               # Bifrost vs MURaM grids and inputs
```

The defaults are the `dense:60` grid, the CPU, the held-out corner and hops of about 190 km (stride 4 on Bifrost, 8 on MURaM). The notebooks (`gnn_inference.ipynb`, `synthesis.ipynb`, ...) use the code of commit fbaba82, and `plot_scripts/plot_graph.py` illustrates the index-space graph of that code.

## 📄 Citation

If you use this code in your research, please cite the original paper describing the method:

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

Please also cite this repository if you use the 3D implementation, and A. Vicente Arévalo et al., A&A, 710, A256 (2026).

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# FUTURE WORK

### 3. Integration with Inversion Codes

This module is designed to act as an "oracle" for inversion codes (like Hazel2 or STiC). Instead of solving the SEE iteratively, the inversion code queries this network:

```python
import torch
from graphnet3d import GNModel

# Load trained model
model = GNModel.load('weights/ca_ii_best.pt')

# Inversion loop
for iteration in range(max_iter):
    # Predict the level populations instantly
    pops = model(current_atmosphere)
    
    # Synthesize spectra using the populations (e.g., with Lightweaver or SIR)
    spectra = synthesize(current_atmosphere, pops)
    
    # ... compute loss and update atmosphere ...

```