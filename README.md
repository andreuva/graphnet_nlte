# GraphNet-3D-NLTE: fast 3D non-LTE Ca II populations with a graph network

A graph neural network that predicts the non-LTE populations of the Ca II model atom (the 4s, 3d and 4p levels and Ca III)
in a 3D solar atmosphere, column by column with the 3D context of the neighbouring columns, from the thermodynamic
state alone. It extends the 1D plane-parallel network of
[Vicente Arévalo et al. (2022)](https://doi.org/10.3847/1538-4357/ac53b3) to 3D: trained on the 3D non-LTE solution of
one Bifrost snapshot, it reproduces the populations of the held-out part of that snapshot to about 0.1 dex end to end and
transfers to a MURaM snapshot at its native resolution. The populations feed a spectral synthesis (Lightweaver here) and
the network is differentiable, so it can act as the non-LTE oracle of an inversion code.

## The model

**Graph.** One graph per predicted column: the 49 columns of a disc of radius 4 inside the 9x9 stencil around it, with a
hop of 1-4 pixels (drawn at random in training, ~190 km at inference), all levels of every column. Star topology: edges
link a node to a node of the central column when their height difference is at most 0.5 Mm (2.0 Mm between two central
nodes; adjacent central levels always), so the receptive field in km is the same for every hop length and vertical grid.
Only the central column is predicted; the neighbours provide the 3D context. Edge features: (dz, horizontal distance,
|dr|) in Mm. Encode-Process-Decode (Battaglia et al. 2018), 8 message-passing steps, latent size 128, 0.92 M parameters.

**Inputs per node (12):** vx, vy, vz in units of 10 km/s; log10 T, log10 rho, log10 P standardized by one global mean and
standard deviation of the training cube; and the six log10 LTE (Saha-Boltzmann) fractions of the Ca levels computed from
the atmosphere's own T and n_e (`normalization.lte_log_fractions`, Lightweaver `lte_pops`). The raw n_e is not an input:
n_e enters only through the Saha dependence, which gives the network the right response to n_e at fixed temperature
(a network with the raw n_e learned Bifrost's equation-of-state correlation and responded to n_e with the wrong sign on
MURaM). No magnetic field, no height coordinate. The statistics are stored in the checkpoint and re-applied to any atmosphere.

**Outputs:** log10 of the six level fractions through a log_softmax head (fractions sum to one exactly). Absolute
populations = fractions x 9.65e14 x rho [kg m^-3] cm^-3, the Ca abundance of the training targets. Loss: MSE on log10 f
weighted by the Ca II relevance of each node (w = max(clip(f_CaII / 1e-2, 0, 1), 0.1)).

**Vertical grids** (`vertical_grids.py`, heights above tau500 = 1, dense where the Ca II lines form, few levels in the
corona). Training samples draw a random two-zone grid from the 425 native levels (60-110 km steps up to a transition at
3.8-4.8 Mm, 5-10 coronal levels, jitter and level removal; bottom and top native levels kept); the per-epoch validation uses
the `dense:60` deployment grid (the three trained seeds were still selected on the 50-level index grid of the original code,
since removed); inference accepts `two_zone[:km]` and `dense[:km]`.
The grid used at inference decides the end-to-end accuracy: on the original 50-level grid every model scores ~0.15 dex at the
native levels because the levels are 170-225 km apart where Ca II ionizes; on `dense:60` (60 km steps from the bottom to
4.5 Mm plus 7 coronal levels, 93 levels) the same models score ~0.10 dex. **Deploy on `dense:60` with ~190 km hops**
(Bifrost stride 4, MURaM stride 8).

**Augmentation:** rotation of the whole volume about the vertical axis (0/90/180/270 degrees, (vx, vy) rotated), nothing
else: no mirrors, no sign flips, nothing vertical.

**Recipe:** AdamW (weight decay 0.01 on the matrices), 1-epoch linear warmup + cosine to zero per batch, gradient clipping
at 1, EMA of the weights (0.9998 with a ramp) validated next to the raw weights, a fresh random 20 % subset of the training
columns every epoch, 15 epochs (~22 min each on one H100; `gpu_memory_fraction` caps the allocator on a shared GPU).
`*_best.pth` holds the weights (raw or EMA, whichever validated better) of the epoch with the lowest held-out RMS dex.

## Recommended model

The three-seed ensemble of `conf.dat` (seeds 777, 778, 779): the mean of log10 n of
`checkpoints/v2_rung8a_lte_seed777/2026.10.16-01:55:25_best.pth`, `..._seed778/2026.10.16-16:19:41_best.pth` and
`..._seed779/2026.10.16-22:43:16_best.pth` on the `dense:60` grid. Bifrost held-out corner, end to end at the 425 native
levels: 0.097 dex RMS (single seeds 0.098-0.101; interpolation floor of the grid 0.073); MURaM photosphere (0-0.5 Mm)
against the SNAPI 1.5D reference: 0.005 dex median (LTE 0.001, the previous model 0.013). The per-node error bar sigma^2 = a^2 + (c s)^2 + F(z, l)^2 (s = seed
spread, F = floor of the grid, a and c per height band in `ensemble.py`) covers 94 % of the Bifrost held-out errors at
2 sigma; on other atmospheres the seed spread is a relative reliability flag. The Ca II 8542 profiles synthesized from the
ensemble populations (`synthesis.py`: Lightweaver, mu 1, no microturbulence) differ from those of the 3D targets by 2.0 % on
average in the +-0.05 nm core (0.28 % over the 1 nm band; core Doppler shifts within 3.5 pm RMS), against 7.8 % (1.4 %) for
a 1D non-LTE solution of each column and 59 % (6.5 %) for LTE populations.

## Data

* **Atmosphere:** Bifrost `en024048_hion`, snapshot 385, 504 x 504 columns of 47.7 km, vertical indices 52-476 of the native
  grid (425 levels, -0.51 to 12.59 Mm; `../en024048_hion/grid_bifrost.npz`), in SI as `../data_train_v2/AR_385_*.dat` with a
  `MANIFEST.json` (shapes, units, sources, checksums), regenerated from the FITS files `en024048_hion/atmos/` by
  `dataset_scripts/make_data_train.py`.
* **Targets:** `AR_385_CaII_5L_pops.dat`, the populations of the five Ca II levels and Ca III in **cm^-3** (the rest is
  SI), a 3D non-LTE solution of the snapshot (attributed to Multi3D; the code version, model atom, CRD/PRD treatment,
  microturbulence and n_e treatment are not documented). The total Ca density equals 9.65e14 x rho[kg m^-3] cm^-3 at every
  point; the network is trained on the level fractions.
* **MURaM:** the enhanced-network snapshot 499000 in `/dat/milic/MURaM_enhanced_network/` (23.4 km pixels, 20 km levels),
  predicted at native resolution, aligned to Bifrost at the column-median tau500 = 1 level; the reference is the SNAPI
  1.5D NLTE solution `che_full_499000_lwsynth_200.0.fits` (populations in HDU2, spectra in HDU0), which agrees with a 3D
  solution only below ~0.5-1 Mm (the 1.5D-vs-3D floor measured on Bifrost is 0.04 dex at 0-0.5 Mm and 0.35-0.6 dex above).

## Installation

```bash
git clone -b new_topologies https://github.com/andreuva/graphnet_nlte.git
conda env create -f environment.yml        # python 3.12, torch 2.7.1 (cu118), torch_geometric 2.6.1, torch_scatter, configobj,
                                           # lightweaver >= 0.15 (the LTE input and the syntheses), astropy, joblib, scipy, matplotlib
```

## Usage

```bash
python gnn_train.py conf.dat [seed]                                   # -> checkpoints/v2_rung8a_lte_seed<seed>/ (last.pth, *_best.pth, training_log.csv, grids.png)
python gnn_inference.py <checkpoint> 4 test cuda:0 dense:60           # Bifrost held-out corner -> results/gnn_inference/<stamp>/ (metrics.json: own grid and native resolution)
python muram_comparison/muram_inference.py <checkpoint> 8 384,640,384,640 cuda:0 dense:60   # MURaM crop, 188 km hops -> results/muram_inference/<stamp>/
python muram_comparison/compare_populations.py <run_dir>              # vs SNAPI 1.5D, with the LTE and climatology baselines
python muram_comparison/ood_table.py <run_dir>                        # MURaM nodes outside Bifrost's input ranges
python muram_comparison/compare_datasets.py [checkpoint] [grid]       # Bifrost vs MURaM on the grid: tick plot of the grids, physical and normalized input distributions (PNGs in the cwd)
python muram_comparison/compare_height_stratification.py [run_dir]    # mean stratifications of Bifrost and MURaM; Ca II populations vs SNAPI (+ the GNN of a run_dir)
python muram_comparison/muram_synthesis.py <run_dir> 8                # Ca II K on a sample of columns (cross-code check)
python ensemble.py <run_dir_seed777> <run_dir_seed778> <run_dir_seed779>   # mean of log10 n, spread, metrics and error bar -> results/ensemble/<stamp>/
python synthesis.py <run_dir>; python plot_scripts/synthesis_plots.py <run_dir>    # Ca II 8542 synthesis of a Bifrost run and its figures
python plot_scripts/inference_plots.py <run_dir>; python plot_scripts/stride_table.py > results/stride_table.md
python muram_comparison/bifrost_15d_floor.py; python muram_comparison/muram_report.py
```
Every script writes `results/<script>/<stamp>/` with a `meta.json` (`provenance.py`). Inference runs on the CPU too
(`cpu` as the device; about 12 ms per column on the original 50-level grid, 3.7 times the edges on `dense:60`). The earlier representations (per-height normalizations,
height and magnetic-field inputs, linear head, 11 inputs) live in the parent commit of this one.

## Citation

If you use this code, please cite the paper describing the method and this repository:

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

## License

MIT, see [LICENSE](LICENSE).
