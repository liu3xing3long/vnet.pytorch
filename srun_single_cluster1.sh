#!/usr/bin/env bash

eps=1000
batch=48
jobs=8
gpus=14

srun -p Single -n1 -w BJ-IDC1-10-10-20-163 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=vnet-lungseg --kill-on-bad-exit=1 \
python train.py --resume "/mnt/lustre/liuxinglong/work/vnet.pytorch/work/luna16_2mm_[4, 5, 5]_20180305_1346/vnet_checkpoint.pth.tar" --ngpu $gpus --batchSz $batch --nEpochs $eps
