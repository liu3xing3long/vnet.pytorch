#!/bin/bash
/home/liuxinglong01/anaconda2/envs/pytorch/bin/python train.py --resume work/vnet_model_best.pth.tar --inference working_imgs --save working_imgs/results
