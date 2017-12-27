#!/bin/bash
python inference.py --resume work/vnet_model_best.pth.tar --inference working_imgs_1mm --save working_imgs_1mm/results
