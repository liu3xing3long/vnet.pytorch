import sys
import os
from os import listdir
from os.path import isfile, join
import torchbiomed.datasets as dset

####################################################
# root_path = 'orig_imgs/'
# target_path = 'working_imgs_1mm_xyz/'

# x = 320
# y = 320
# z = 320
# spacing = 1.0


####################################################
# root_path = 'orig_imgs/'
# target_path = 'working_imgs_1mm/'
#
# x = 320
# y = 320
# z = 256
# spacing = 1.0


####################################################
root_path = 'orig_imgs/train_subset00/'
target_path = 'working_imgs_2mm/'

x = 192
y = 192
z = 160
spacing = 2


####################################################
dset.luna16.normalize_lung_CT(src=root_path,
                              X_MAX=x, Y_MAX=y, Z_MAX=z, vox_spacing=spacing,
                              dst=target_path)
