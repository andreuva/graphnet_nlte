#!/usr/bin/env bash
# README steps 4a and 4b on one run directory. Run from inside graphnet_nlte/:
#   ./run_tests.sh ./checkpoints_si_v3/<run>/ [gpu] [database dir]
set -e

: "${1:?usage: $0 <run_dir> [gpu=0] [database_dir=../data_1d_si_v3/]}"
RUN=${1%/}/                  # test_prediction.py needs the trailing slash
GPU=${2:-0}
RD=${3:-../data_1d_si_v3/}

python test_prediction.py --dtst validation --gpu "$GPU" --rd "$RD" --sav "$RUN" --testdir "$RUN"
PKL=$(ls -t "$RUN"validation_checkpoint_*.pkl | head -1)   # the pickle just written
python plot_scripts/explore_tests.py --ck "$PKL"
python evaluate_intensity.py --rd "$RD" --pred "$PKL" --n 1000
