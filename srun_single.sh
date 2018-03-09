#!/usr/bin/env bash

eps=400
batch=48
jobs=1
gpus=8

srun -p Med -n1 -w BJ-IDC1-10-10-15-73 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=vnet-lungseg --kill-on-bad-exit=1 \
python train.py --ngpu $gpus --batchSz $batch --nEpochs $eps
