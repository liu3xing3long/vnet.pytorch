#!/bin/bash
python inference.py --ngpu 4 --resume work/vnet_model_best.pth.tar --infertype 0 --inference ./working_imgs_1mm_xyz --save ./working_imgs_1mm_xyz/results
