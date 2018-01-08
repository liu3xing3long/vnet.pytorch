#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:29:46 2017

@author: liuxinglong01
"""
import sys
import os
from os import listdir
from os.path import isfile, join
import torchbiomed.datasets as dset

root_path = '/home/liuxinglong01/1HDD/data/LUNA'

##########################################################
target_path = 'luna16_1mm_xyz'
x = 320
y = 320
z = 320
spacing = 1.0

##########################################################
# target_path = 'luna16_1mm'
# x = 320
# y = 320
# z = 256

##########################################################
# target_path = 'luna16_2mm'
# x = 160
# y = 160
# z = 128
# spacing = 2


##########################################################
def GetFileFromThisRootDir(dir,ext = None):
    allfiles = []
    needExtFilter = (ext != None)
    for root,dirs,files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def getFileName(path):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    files = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.mhd':
            #print i
            files.append(path + '/'+ i)
    return files


##########################################################
# files = getFileName(root_path)
# dset.luna16.build_nodule_offset_tables(files, 'luna16/tmp-nodule-tables',
#                                        X_MAX=x, Y_MAX=y, Z_MAX=z, vox_spacing=spacing)
# dset.luna16.normalize_nodule_mask(orig='luna16/', src='luna16/tmp-images/',
#                                   X_MAX=x, Y_MAX=y, Z_MAX=z, vox_spacing=spacing,
#                                   dst='luna16/normalized_nodule_masks/',
#                                   tables='luna16/tmp-nodule-tables/',
#                                   annotations='luna16/annotations.csv')
# dset.luna16.normalize_lung_CT(src='luna16/tmp-images/',
#                               X_MAX=x, Y_MAX=y, Z_MAX=z, vox_spacing=spacing,
#                               dst='luna16/normalized_ct_images/')
# dset.luna16.normalize_lung_mask(src='luna16/tmp-masks/',
#                                   X_MAX=x, Y_MAX=y, Z_MAX=z, vox_spacing=spacing,
#                                  dst='luna16/normalized_lung_masks/')


##########################################################
for subset in range(10):
    dset.luna16.normalize_lung_CT(src="{0}/{1}/{2}/{3}/".format(root_path, "original_lungs", "subset", subset),
                                  X_MAX=x, Y_MAX=y, Z_MAX=z, vox_spacing=spacing,
                                  dst=target_path + "/normalized_ct_images/")
dset.luna16.normalize_lung_mask(src="{0}/{1}/".format(root_path, "seg-lungs-LUNA16-orig"),
                                X_MAX=x, Y_MAX=y, Z_MAX=z, vox_spacing=spacing,
                                dst=target_path + '/normalized_lung_masks/')