# graphnet_nlte

Graph networks for solving radiative transfer problems in stellar atmospheres.

Andres Vicente & Andrés Asensio:
[Accelerating non-LTE synthesis and inversions with graph networks](https://arxiv.org/pdf/2111.10552.pdf)

The current configuration trains a graph network to predict the 16 departure coefficients
`b_i = n_i / n_i*` of a Si I/II/III model atom from a 1D atmospheric column, so that the
Si I 1083.0 nm line can be synthesised with a single formal solution instead of a full
statistical-equilibrium iteration.

---

## What is in here

| File | Role |
|---|---|
| `generate_database.py` | MPI database generation: perturbs model atmospheres, solves NLTE with lightweaver, writes the features and targets |
| `dataset_scripts/clean_dataset.py` | Removes failed samples from a generated database, keeping every file in sync |
| `Dataset.py` | Builds the graphs: node/edge features, normalisation, targets, loss mask |
| `graphnet.py` | The Encode-Process-Decode network |
| `Formal.py` | Training / validation / test loops, checkpointing |
| `train.py` | Training entry point |
| `test_prediction.py` | Runs a checkpoint over a split and dumps predictions vs. targets |
| `plot_scripts/explore_tests.py` | Plots those predictions and the profiles they imply |
| `evaluate_intensity.py` | **Acceptance test**: how much closer to the truth the network gets than LTE, in intensity |
| `api.py` | Minimal interface for calling the trained network from an inversion code |
| `si_atom.py` | The custom 16-level Si I/II/III model atom (not in released lightweaver) |
| `conf.dat` | Network hyperparameters |

The steps below assume you run everything from inside `graphnet_nlte/`, with the databases one
level up (`../data_1d_si_v3/` and so on). Nothing depends on that layout — every path is a flag.

---

## 0. Setup

The environment is in `environment.yml`:

    conda env create -f environment.yml
    conda activate gph

It resolves from conda-forge plus PyPI wheels for PyTorch (official CUDA 11.8 build, which is
what the development machine runs) and `lightweaver`. See the header of `environment.yml` for
the CPU swap and for building `mpi4py` against a site MPI instead.

To build one by hand instead, the packages that matter are `lightweaver`, `pytorch`,
`pytorch_geometric`, `mpi4py`, `configobj`, `scipy`, `scikit-learn`,
`astropy`, `numpy`, `matplotlib` and `tqdm`.

The 16-level Si I/II/III model atom this project is built on is **not** part of any released
lightweaver -- it started as a local patch to `lightweaver/rh_atoms.py`. It now lives in
`si_atom.py` in this directory, so import it from there rather than from `lightweaver`:

    from si_atom import Si_atom_custom

### Input data

Database generation needs **one directory** containing both:

* the semi-empirical reference atmospheres, `*.atmos` in MULTI format (FAL A/C/F/X-CO, the
  `model100X` series, `T5750_g4.5_p00_ext`) — 16 models in the current set;
* the two Bifrost snapshots, `snap385_rh.save` (train and test) and `snap530_rh.save`
  (validation), as IDL save files.

In this installation that directory is `../data_1d_old/data/models_atmos/`. The originals, along
with an already-generated database and pretrained checkpoints, are at
[data and pretrained models](https://cloud.iac.es/index.php/s/JR3GQym9mgNk4mL).

---

## 1. Generate the database

Three separate runs, one per split. They are independent and can be run one after another:

    # training split  (Bifrost snap385, first 80% of the cube in x)
    mpiexec -n 48 python generate_database.py --train  1 --n 500000 --f 20000 \
        --rd ../data_1d_old/data/models_atmos/ --sav ../data_1d_si_v3/

    # test split      (Bifrost snap385, remaining 20% in x)
    mpiexec -n 48 python generate_database.py --train  0 --n 120000 --f 20000 \
        --rd ../data_1d_old/data/models_atmos/ --sav ../data_1d_si_v3/

    # validation split (Bifrost snap530, a different snapshot entirely)
    mpiexec -n 48 python generate_database.py --train -1 --n  25000 --f 20000 \
        --rd ../data_1d_old/data/models_atmos/ --sav ../data_1d_si_v3/

| Flag | Meaning |
|---|---|
| `--train` | `1` train, `0` test, `-1` validation. Also sets the output prefix. |
| `--n` | Number of samples to compute |
| `--f` | Write a checkpoint of the database every this many *completed* samples |
| `--rd` | Input directory (see above) |
| `--sav` | Output directory; created if missing |
| `--prd` | Partial redistribution, `0` (default) for this line |
| `--seed` | Order in which each split consumes its own Bifrost columns, and every perturbation of the reference atmospheres (default 1234) |

`mpiexec -n K` uses one master and `K-1` workers, so ask for one more process than the number of
solves you want in flight. This machine has 192 cores. Each NLTE solve takes roughly 0.2–1.5 s
of single-core CPU, so 500k training samples on 48 processes is a few hours.

**Choosing `--n`.** Each sample is drawn 50/50 from a Bifrost column or a perturbed reference
atmosphere, until the Bifrost columns for that split run out; after that everything comes from
the reference atmospheres. The 504×504 cube gives 203,112 columns to train and 50,904 to test,
so `--n` of about 2× those numbers consumes all of them. For reference, the previous database
came out as:

| split | samples | Bifrost | reference-derived |
|---|---|---|---|
| train | 493,398 | 203,212 | 290,186 |
| test | 98,873 | 50,804 | 48,069 |
| validation | 19,763 | 10,316 | 9,447 |

**Choosing `--f`.** Each checkpoint rewrites the whole database, and a full training split is
~22 GB, so `--f 10` (the default) would spend the entire run on I/O. Something around
`--n / 25` is a reasonable trade between lost work on a crash and time spent writing.

### Output

Per split, ten pickles named `<split>_<quantity>.pkl`:

`T`, `z`, `ne`, `vturb`, `vlos`, `tau` (the atmosphere), `logdeparture` (the target,
log₁₀(n/n\*)), `n_Nat` (log₁₀(n/n_total), used to build the loss mask), `Iwave` (the reference
emergent profile) and `wave`.

All of them are lists with one entry per sample **except `wave`**, which is the single shared
wavelength grid the `Iwave` arrays live on — that is why `clean_dataset.py` leaves it alone.

### How the splits are built

`--train 1` and `--train 0` read the same snapshot and partition it by *position*: training takes
the first `TRAIN_X_FRACTION` (0.8) of the cube's x-extent, test takes the rest. The partition is
deterministic, so the two runs are always disjoint, and the boundary is one line through the
snapshot rather than a scatter of pixels — neighbouring columns of a granulation snapshot are
strongly correlated, so a random per-column split leaves near-copies of training columns in the
test set even when it is a correct partition. `--seed` only shuffles the order in which a split
consumes its own columns, so a run stopped early is still a reproducible, representative sample.

`--train -1` reads `snap530_rh.save` instead and uses all of it, which makes validation the
cleanest of the three splits: a different snapshot, sharing nothing with either of the others.

Three details worth knowing about the atmospheres themselves:

* Bifrost columns carry no microturbulence of their own, while every reference atmosphere
  does; with `vturb = 0` that one feature identified the data branch exactly. Each Bifrost
  column is therefore given the (perturbed) `vturb` stratification of a randomly chosen
  reference atmosphere, interpolated in height, before the NLTE solve.
* The reference-atmosphere branch hands lightweaver `ne=None` and lets it reconstruct hydrostatic
  equilibrium, which invents both `ne` and `nHTot`; only `ne` is stored. The atmosphere is
  therefore rebuilt from that reconstructed `ne` before the NLTE solve, so the column that is
  solved is exactly the one a consumer rebuilds from the stored features (`api.py` passes `ne`
  back into `make_1d`, which derives `nHTot` from the electron pressure). Without that rebuild
  the stored targets belong to an atmosphere the feature vector cannot reproduce, by up to ~1 dex.
* The emergent intensity is evaluated at `atmos.muz[-1]` = 0.9531, i.e. θ = 17.6°, the outermost
  node of the 5-point quadrature — not at disk centre.

---

## 2. Clean the database

Samples whose NLTE solve fails are stored as `None` and rescheduled, but a run that ends while
some are outstanding leaves holes. Remove them from every file at once:

    python dataset_scripts/clean_dataset.py --dir ../data_1d_si_v3/

It finds every prefix in the directory, takes the union of the indices that are `None` or
non-finite in *any* file, and drops those indices from *all* of them, so the files stay aligned.
It rewrites in place — copy the directory first if you want to keep the raw output.

---

## 3. Train

    python train.py --epochs 300 --batch 64 --lr 1e-4 --gpu 0 \
        --conf conf.dat --rd ../data_1d_si_v3/ --sav ./checkpoints_si_v3/

This trains on `<rd>/train_*.pkl` and validates on `<rd>/validation_*.pkl` (the snap530
split), so the loss that selects the best checkpoint is measured on a genuine hold-out. An
earlier version carved validation out of `train_*` at random; because neighbouring Bifrost
columns are near-copies of each other, that number read about 2× lower than the true hold-out
loss for the same checkpoint. Other flags: `--seed` (parameter initialisation and batch order,
default 0), `--compile` (`torch.compile`, about 5 min of compilation for a ~1.5× faster step),
`--smooth` (training-loss display only) and `--conf` (the hyperparameters).

`train.py` creates a timestamped run directory under `--sav`, copies `conf.dat`, `Dataset.py`,
`Formal.py`, `graphnet.py` and `train.py` into it, and keeps two checkpoints there:
`best.pth`, overwritten whenever the validation loss improves (weights only), and `last.pth`,
overwritten every epoch with the optimizer and scheduler state as well. To continue an
interrupted run in place:

    python train.py --resume ./checkpoints_si_v3/<run>/ --epochs 300 --batch 64 --lr 1e-4 --gpu 0 --rd ../data_1d_si_v3/

Each checkpoint carries its hyperparameters and the exact normalisation constants it was
trained with, so it can always be reproduced later.

The learning rate warms up linearly over the first epoch and then follows a cosine decay to
zero, stepped per batch. The processor uses pre-norm residual blocks (LayerNorm on the inputs of
each message-passing step, and one before the decoder); the previous post-norm layout let the
latent norm grow ~8× over the 100 steps and the validation loss oscillated by up to 2× between
epochs.

Network size is set by `conf.dat`:

    node_input_size = 5          # log10 T, z, log10 ne, vturb, vlos
    edge_input_size = 1          # delta z between neighbouring depth points
    global_input_size = 1
    latent_size = 128
    mlp_hidden_size = 128
    mlp_n_hidden_layers = 3
    n_message_passing_steps = 100
    output_size = 16             # one departure coefficient per level

`n_message_passing_steps` is the one to think about: the graph is a nearest-neighbour chain, so
after K steps a node has seen exactly K depth points either side. Bifrost columns have 211
points, so K = 100 couples about half a column.

### About the loss

The loss is a **masked** MSE. Levels and depths carrying less than
`Dataset.NEGLIGIBLE_LOG_N_OVER_NTOT` (10⁻⁹) of the species population are dropped from it. They
are physically inert — they move the emergent 1083 nm profile by less than a part in 10⁶ — but
numerically loud: about 16% of the raw targets sit on the ±10 clip plateau at |y| = 2 after
scaling, while the whole line-forming region lives inside |y| ≤ 0.2. At this threshold 20% of
the points are masked and 83% of the clip plateau goes with them. The mask is built from the
`*_n_Nat.pkl` files; without them nothing is masked and the loss is the plain MSE.

Because the network is unsupervised on the masked points, `api.compute_dep_coeffs` clamps its
output to the same ±10 the targets are clipped to.

Two consequences for reading the numbers:

* **the reported MSE is not comparable to runs from before masking** — it will read higher;
* the MSE is only a proxy in any case. The validation loss is a true mean over the whole
  `validation_*` split (so `best.pth` is selected on a stable number), but whether a checkpoint
  actually improves the *profiles* is what step 4b measures.

---

## 4. Test

### 4a. Predictions against a held-out split

    python test_prediction.py --dtst validation --batch 64 --gpu 0 \
        --rd ../data_1d_si_v3/ --sav ./checkpoints_si_v3/<run>/ --testdir ./checkpoints_si_v3/<run>/

Note the flag names: `--sav` is where the **checkpoint is read from** and `--testdir` is where
the **result is written**. `--sav` must be a single run directory *with a trailing slash*; its
`best.pth` is used (older runs with several `<stamp>_best.pth` files: the last one), and the
search is not recursive.

This writes `<dtst>_checkpoint_<stamp>.pkl` holding the predictions, the targets, the normalised
input features and the loss per batch.

Use `--dtst validation`. A `test` split generated by the current code is a genuine hold-out, but
any database generated before the spatial-split fix shares ~80% of its Bifrost columns with
`train`, so a number measured on one of those is a memorisation score.

Then plot 25 random columns and the profiles they imply:

    python plot_scripts/explore_tests.py --ck ./checkpoints_si_v3/<run>/

`--ck` takes a run directory, a whole checkpoint tree (every run below it), a `*.pth` file or a
prediction pickle; the database is read from the directory recorded in each pickle (`--rd`
overrides it). It writes the loss curves, a grid of departure coefficients and the corresponding
Si I profiles into `<run>/plots/` (`--sav` overrides), and with more than one pickle also a
loss-vs-architecture scatter.

### 4b. Acceptance test in intensity

This is the number to quote. `evaluate_intensity.py` synthesises the Si I 1083.0 nm profile four
times per column — a fully converged NLTE solve (the truth), the stored departure coefficients
with a single formal solution, the network's coefficients with a single formal solution (the
deployed path of `api.intensity_gnn`), and LTE — and measures the network where an inversion
would feel it:

    # run the checkpoint through the deployed feature construction
    python evaluate_intensity.py --rd ../data_1d_si_v3/ --ck ./checkpoints_si_v3/<run>/ --n 1000

    # or evaluate exactly the predictions test_prediction.py dumped
    python evaluate_intensity.py --rd ../data_1d_si_v3/ \
        --pred ./checkpoints_si_v3/<run>/validation_checkpoint_<run>_at_<stamp>.pkl

The report (printed, and written to `<run>/acceptance/` with a JSON of headline numbers, a pickle
of every per-column row and six figures) contains:

* an **error budget** — RMS of `(I - I_conv)/I_c` for the deployed network, the network alone,
  the J = 0 formal-solution floor and LTE, as median with bootstrap CI, p90, p99 and max;
* an **observational-noise test** — fraction of columns whose error is below 1e-3 and 3e-4 `I_c`
  and whose reduced chi^2 against the truth is <= 1 at that noise;
* **line parameters** an inversion would infer — line depth, equivalent width, core velocity and
  FWHM — with the skill score `1 - GNN/LTE` and the paired win fraction;
* **departure coefficients weighted by the line-core contribution function** — `log10 b_lower`
  (opacity) and `log10 b_u/b_l` (source function), the loss-mask MAE, population conservation and
  the shift of the tau = 1 height;
* results **per column origin** (Bifrost vs semi-empirical) and per line depth, the worst columns
  by index, the **speed-up** of the NLTE step and end to end, and consistency checks proving the
  converged solve here reproduces the database.

Synthesis runs on `--workers` CPU processes (about 1.5 min for 1000 columns on 32); the network
runs on `--device` (CPU by default, which is plenty for evaluation).

The last line is a diagnostic, not an error: the 16 levels are decoded independently, so the
predicted populations do not conserve the Si total the way the lightweaver targets do.

---

## 5. Inference from your own code

`api.py` is the interface for an external inversion. Four functions, all in SI units, with `z`
**strictly decreasing** (index 0 = top of the atmosphere):

```python
import sys; sys.path.insert(0, '/path/to/graphnet_nlte')
import api

# GraphNet only, one column, no lightweaver. ~170 ms on CPU, ~50 ms on a GPU: the network is
# 600 layers deep and a single column cannot fill a GPU, so this is launch overhead.
log_dep = api.compute_dep_coeffs(T, z, ne, vturb, vlos)

# GraphNet only, many columns in ONE forward pass. ~1.3 ms per column on an H100 at 64 columns.
# This is the call for the hot loop of an inversion: batch every pixel of an iteration.
log_deps = api.compute_dep_coeffs_batch([(T, z, ne, vturb, vlos), ...], device='cuda:0')

# GraphNet + one formal solution. Drop-in replacement for the full solve.
wave, Iwave, log_dep = api.intensity_gnn(T, z, ne, vturb, vlos)

# Full lightweaver NLTE solve. Ground truth, orders of magnitude slower.
wave, Iwave, log_dep = api.synthesis_lw(T, z, ne, vturb, vlos)
```

`api.DEFAULT_CHECKPOINT` points at the `checkpoints_si_v2/` tree and resolves to the most recent
`best.pth` below it on every call, so it follows training automatically; pass `checkpoint=`
explicitly to pin a model. Point it at your new tree once step 3 is running.

Two limits to be aware of:

* `intensity_gnn` takes a single formal solution on a Context that has never been iterated, so
  J = 0 and the background coherent-scattering emissivity is missing. That is accurate in the
  near-IR (~10⁻⁴ relative against the converged solve) but not in the UV, where it reaches ~0.7
  at 100–200 nm. The function warns below 400 nm; use `synthesis_lw` there.
* `DEFAULT_WAVE` samples 1074–1085 nm at 2 pm, which keeps at least two samples across the Si
  Doppler width (4.4 pm at 2500 K) everywhere in the training set. If you pass your own `wave`,
  keep it at least that fine or the line core will read shallow.

---

## Known limitations

Open items, in the order they are worth addressing:

* **Populations are not conserved.** Σ nᵢ ≠ n_total for the predicted populations, by ~2–5% in
  the photosphere, which is a systematic floor on the line depth. Renormalising in
  `compute_dep_coeffs` is cheap and helps; training on `n_Nat` instead of `logdeparture` would
  remove it properly.
* **The receptive field is shorter than a column.** See `n_message_passing_steps` above.
* **A third of the generated atmospheres have no photosphere.** Perturbing T by σ = 2500 K at
  every knot and re-integrating hydrostatic equilibrium moves the bottom density by orders of
  magnitude: ~35% of columns never reach τ₅₀₀ = 1 and ~11% exceed 10⁶. They are valid,
  converged atmospheres, just not ones the Si I 1083 nm line is ever inverted in, and they
  stretch the input normalisation. Rejecting on max τ₅₀₀ at generation time would fix it.
* **The stored `tau` column is not a calibrated depth scale.** lightweaver reports
  τ₅₀₀ = 2.5×10⁻³ at FALC's own h = 0, where the model defines it to be ~1. Nothing in the
  current configuration reads it — the network uses `z`, and the perturbation knots no longer
  depend on it — but do not build anything on it without checking first.

`tests/` holds scripts from the earlier Ca II work. They do not run against the current pipeline
and are kept for reference only.
