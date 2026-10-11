# GraphNet-3D-NLTE: Fast 3D Non-LTE Atomic Populations with Graph Neural Networks

**GraphNet-3D-NLTE** is a deep learning framework designed to accelerate the computation of non-LTE (Local Thermodynamic Equilibrium) atomic populations in 3D solar and stellar atmospheres.

Building upon the 1D architecture proposed by [Vicente Arévalo et al. (2022)](https://doi.org/10.3847/1538-4357/ac53b3), this project extends the concept to fully **3D geometries**. It utilizes Graph Neural Networks (GNNs) to predict the atomic level **populations** (not departure coefficients) orders of magnitude faster than traditional 3D radiative transfer solvers.

## 📖 Table of Contents

* [Background & Motivation]()
* [Key Features]()
* [Installation]()
* [Dataset Preparation]()
* [Usage]()
* [Architecture (From 1D to 3D)]()
* [Performance]()
* [Citation]()

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

* **Nodes:** Same physical properties, but representing a voxel.

* **Edges:** Connections are defined by a 3D stencil (e.g., k-nearest neighbors or a N-point stencil). Encodes the 3D geometric distance between voxels.

* **Message Passing:** Information propagates primarily vertically (due to stratification) but also horizontally, mimicking 3D radiative transfer effects, and only to the center column, neibourgs are only used for context and not actually predicted.

## ✨ Key Features

* **3D Spatial Awareness:** unlike 1D column-by-column approaches, this code treats the atmosphere as a 3D connected graph, accounting for horizontal radiative transfer effects indirectly through learned spatial correlations.

* **Arbitrary Grids:** The Graph Network architecture () allows for variable grids. Nodes can be separated different horizontal distances, better generalizing to different resolution datasets.

* **Differentiable:** The network is fully differentiable, allowing seamless integration into inversion codes for calculation of Response Functions via backpropagation.

* **Speed:** Predicts level populations in milliseconds, offering a large speedup over traditional solvers.

## ⚙️ Installation

Clone the repository and install the dependencies. A GPU is strongly recommended for inference and unavoidable for training.

```bash
git clone https://github.com/andreuva/graphnet_nlte/tree/new_topologies
cd graphnet_nlte
```

**Key Dependencies:**

* `torch` & `torch_geometric` (for Graph Network implementation )
* `conda` for environment management
* `cuda` (if using GPU acceleration)

## 💾 Dataset Preparation

The model requires training data consisting of model atmospheres and their corresponding "ground truth" populations computed by a specific non-LTE code (e.g., Multi3D, PORTA, or other).

1. **Input:** 3D MHD snapshots (e.g., Bifrost ).

2. **Target:** Normalized populations calculated via a trusted radiative transfer code.

**Data Structure:**
The code expects data in BIFROST format, but easy adaptation to other formats is possible:

* **Nodes ():** Features 
 
* **Edges ():** Connectivity indices and edge features (e.g., geometric distance).

### Training data used on this branch (`new_topologies`)

* **Atmosphere:** Bifrost `en024048_hion`, snapshot 385, 504 x 504 columns, vertical indices 52-476 of the native grid (425 levels, -0.51 to 12.59 Mm), stored in SI units as `data_train/AR_385_{temp,veloc,B,ne,mass,press}.dat` (produced by `bifrost_raw_read_test.ipynb` from `en024048_hion/atmos/BIFROST_en024048_hion_*_385.fits`). Known defect: the committed `AR_385_B.dat` has Bz == By (the notebook read `by` twice); corrected in `data_train_v2/`.
* **Targets:** `data_train/AR_385_CaII_5L_pops.dat`, the populations of the 5 Ca II levels plus Ca III, in **cm^-3** (the rest of the data is SI). The network is trained on log10 of the level *fractions*; the total Ca density equals 9.65e14 * rho[kg m^-3] cm^-3 at every point.
* **Provenance of the targets, as far as it is known:** a 3D non-LTE solution, attributed to Multi3D in `stride_tests.ipynb`; `porta_scripts/caiigen.c` reads this file as *input*, so it is not PORTA output. The code version, model atom, CRD/PRD treatment, microturbulence and the n_e treatment used to compute them are **not documented**.


## 🚀 Usage

### Representation (v2, since 2026-10)

* **Inputs per node (7):** vx, vy, vz in units of 10 km/s; log10 T, log10 n_e, log10 rho, log10 P standardized by one global
  mean and standard deviation of the training cube (clipped at +-5 sigma). No magnetic field, no height coordinate.
  The statistics are stored in the checkpoint and re-applied to any other atmosphere (`normalization.py`).
* **Edges:** star topology around the central column; features (dz, horizontal distance, |dr|) in Mm. Two rules
  (`[dataset] edge_rule`): `index` (ladder rungs 0-4: radius in index units, the vertical reach grows with the stride) and
  `physical` (rung 5a: fixed vertical reach in Mm, the same receptive field for every stride and grid).
* **Vertical grid** (`[dataset] grid`, `vertical_grids.training_grid`): `index` (30 + 20 levels in native index) or `two_zone`
  (rung 6: 80 km steps from tau500 = 1 to 4.5 Mm plus 7 coronal levels, defined in physical height for Bifrost and MURaM alike).
* **Outputs:** log10 fractions of the 6 Ca levels through a log_softmax head (fractions sum to one exactly); loss = MSE on
  log10 f weighted by the Ca II relevance of each node. Absolute populations = fractions x 9.65e14 x rho [kg m^-3] cm^-3.

The per-height normalizations, the height input, the magnetic field and the linear head of the earlier models were removed
from the code after commit 4b1ebbf (check it out to re-run those configurations).

### 1. Training

Configure `conf.dat` (the default is rung 4 of the v2 ladder; `conf_v2_rung*.dat` document the ladder) and run

```bash
python gnn_train.py [conf.dat]
```

Every epoch writes `last.pth`, `training_log.csv` and, on improvement of the held-out RMS dex, a `*_best.pth` with the
configuration, normalization statistics, metadata and git state inside (`checkpoints/<savedir>/`).

### 2. Evaluation on Bifrost

```bash
python gnn_inference.py <checkpoint> <stride> <split> <device> [zero_features]   # results/gnn_inference/<stamp>/
python synthesis.py [run_dir]                                                     # Ca II 8542 synthesis of a run
python plot_scripts/stride_table.py > results/stride_table.md
```

### 3. MURaM

```bash
python muram_comparison/muram_inference.py <checkpoint> <stride> <crop|full> <device>   # results/muram_inference/<stamp>/
python muram_comparison/compare_populations.py [run_dir]; python muram_comparison/ood_table.py [run_dir]
python muram_comparison/muram_synthesis.py [run_dir] [n_side]; python muram_comparison/muram_report.py
```

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

Please also cite this repository if you use the 3D implementation.

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

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
    # Predict level populations instantly
    pops = model(current_atmosphere)
    
    # Synthesize spectra using the populations (e.g., with Lightweaver or SIR)
    spectra = synthesize(current_atmosphere, pops)
    
    # ... compute loss and update atmosphere ...

```