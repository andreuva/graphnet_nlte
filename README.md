# GraphNet-3D-NLTE: Fast 3D Non-LTE Departure Coefficients with Graph Neural Networks

**GraphNet-3D-NLTE** is a deep learning framework designed to accelerate the computation of non-LTE (Local Thermodynamic Equilibrium) atomic populations in 3D solar and stellar atmospheres.

Building upon the 1D architecture proposed by [Vicente Arévalo et al. (2022)](https://doi.org/10.3847/1538-4357/ac53b3), this project extends the concept to fully **3D geometries**. It utilizes Graph Neural Networks (GNNs) to predict departure coefficients ($\beta$) orders of magnitude faster than traditional 3D radiative transfer solvers.

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

* **Speed:** Predicts departure coefficients in milliseconds, offering a speedup factor of  over traditional solvers.

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


## 🚀 Usage

### 1. Training

To train a new 3D model (e.g., for Calcium II), configure the parameters in `conf.dat` (self explanatory fields) and run:

```bash
python gnn_train.py

```

The training process optimizes the Mean Squared Error (MSE) between the predicted and computed populations normalized properly.

### 2. Inference (Prediction)

Generate departure coefficients for a new 3D snapshot:

```
gnn_inference.ipynb
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
    # Predict Departure Coefficients (b) instantly
    b_coeffs = model(current_atmosphere)
    
    # Synthesize spectra using b (e.g., with Lightweaver or SIR)
    spectra = synthesize(current_atmosphere, b_coeffs)
    
    # ... compute loss and update atmosphere ...

```