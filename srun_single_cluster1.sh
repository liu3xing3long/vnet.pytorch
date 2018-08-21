#!/usr/bin/env bash

eps=4000
batch=48
jobs=8
gpus=8

srun -p Single -n1 -w BJ-IDC1-10-10-20-157 --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=vnet-lungseg --kill-on-bad-exit=1 \
python train.py --resume "/mnt/lustre/liuxinglong/work/vnet.pytorch/work/luna16_2mm_[4, 5, 5]_20180319_0642/vnet_checkpoint.pth.tar" --ngpu $gpus --batchSz $batch --nEpochs $eps
