#!/usr/bin/env bash

gpus=14

srun -p Single -n1 -w BJ-IDC1-10-10-20-163 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=vnet-lungseg --kill-on-bad-exit=1 \
python prepare_data.py