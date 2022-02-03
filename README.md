# graphnet_nlte
Graphnets for solving radiative transfer problems in stellar atmospheres.

Andres Vicente & Andrés Asensio:
[Accelerating non-LTE synthesis and inversions with graph networks](https://arxiv.org/pdf/2111.10552.pdf)

## Dependencies

It is recommended to create a new `conda` environment to run this experiment.
To do this, first download the latest version of miniconda at [Conda webpage](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links:)
and follow the instructions of your particular system [install instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
Once you have conda install you should activate conda envioroment as:

     source /$install_directory$/miniconda3/bin/activate

And then you can create and activate a new environment to run the experiment as:

     conda create -n graphnet python=3.8
     conda activate graphnet

You will need to install the following packages for running the network:

     pip install lightweaver
     conda install  mpi4py configobj pytorch cudatoolkit openmpi
     pip install --upgrade numpy
     pip install --upgrade numba

This implementation of Graphnet also depends on the PyTorch Geometric package.
Check the webpage for more specific, but here you can find the usual installation:

     conda install -c conda-forge pytorch_geometric
     conda install scipy

## Data
Now you should be able to run the database generation scripts. To
generate the databases you will need to have the model atmospheres (FALC+Biforst).
You can find all the required data for that, as well as an already generated dB,
the checkpoints of a pretrained GN and the tests for that GN with that checkpoint at:
[data and pretrained models](https://cloud.iac.es/index.php/s/JR3GQym9mgNk4mL)
The recommended folder structure should be like this (other arrangements are also possible):

```bash
.
├── checkpoints
│   ├── crd
│   └── prd
├── graphnet_nlte
│   ├── dataset_scripts
│   ├── plot_scripts
│   └── tests
├── data
│   ├── crd
│   ├── models_atmos
│   └── prd
└── tests
    ├── crd
    │   └── plots
    └── prd
        └── plots
```

### Pretrained and Precomputed models

The data provide not only comes with the model atmospheres, you can also find there the precomputed
databases for the PRD and CRD cases (train/validation/test) and one checkpoint of the PRD and CRD
networks. The databases can be used to train with different hyperparameters (just changing the `conf.dat`)
and the pretrained models can be used for inference as well as test it in other database specifying it
at runtime with the `--readir=../data/other_db/` flag.

## Database generation
At this point to generate the database just go to graphnet_nlte dir and run:

    mpiexec -n 4 python generate_database.py --n 1000 --freq 100 --readir ../data/models_atmos/ --savedir ../data/database/ --train 1 --prd 0

This will generate a training database at `../data/database/` with 1000 models in CRD, saving
the database every 100 samples and using 4 cores.
If you have placed the model atmospheres folder `models_atmos` somewhere else, 
or you have different models, you can specify where to look for with the --readir flag.

This is a computationally heavy procedure that is MPI parallelized. It will generate a
few files containing temperature stratifications, column mass, and optical depths, as
well as departure coefficients.
If your machine has more cores you can take advantage by increasing the used cores
with the -n $ncores flag. It may take a while so you can go to make some coffee :).


## Graphnet training

Once you have the data (either generated or used the provided one) to train with,
you can train the network. The configuration of the Graphnet model is tuned with
a configuration file, that needs to be passed to the training script. 
An example is given by `conf.dat`, so that training can be done using:

    python train.py --epochs=100 --sav ../checkpoints/savedir/ --readir ../data/database/ --batch 32

Where `../data/database` should contain the dataset created earlier (change for `../data/crd`
to use the provided one). This will train the network for 100 epochs. You can
also change the parameters of the training, for example, the configuration file `--conf=conf.dat`,
the validation split `--split=0.2`, the GPU you want to run the training in `--gpu=0`,
the LR `--lr=1e-4`, or the smoothing factor `--smooth=0.05`.


## Verification

The results of the training can be checked against the validation dataset with: 

    python test_prediction.py --readir ../data/database/ --sav ../checkpoints/savedir/ --testdir ../test/test_dir/

This will read the data in the `../data/dataset/validation_*` and save a pickle object in
`../test/test_dir/` folder with the results of the prediction. Note that the dataset is
not the same as for training so you will need to have generated another dataset with
the validation flag `--train -1`. If you want to change the network to test, just change
the `--sav ../checkpoints/other` to other GN checkpoints.
Then with those results, you can visualize 25 random predictions and intensity profiles 
from each of the testing files with:

    cp plot_scripts/explore_tests.py .
    python explore_tests.py
    rm explore_tests.py

This will save the plots inside the same directory in which the test has been saved.

## Inference

To do test inference of model atmosphere you can run the scripts `tests/predict_simple.py`.
Note that this implementation is just for testing the network and has a huge overhead when
loading the data and other unnecessary tasks. For a more efficient implementation to do inversions please
contact the authors.
