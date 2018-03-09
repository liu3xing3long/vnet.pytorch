#!/bin/bash
python inference.py --ngpu 1 --resume work/vnet_model_best.pth.tar --infertype 1 --inference ./working_imgs_2mm --save ./working_imgs_2mm/results
