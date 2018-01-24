#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:19:21 2018

@author: liuxinglong01
"""

import numpy as np
import SimpleITK as sitk


# data = '/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/results/LKDS-00065.mhd'
data = '/home/liuxinglong01/1HDD/work/chest_segmentation/results/seg/LKDS-00193.npz.mhd'

img = sitk.ReadImage(data)

lmap = sitk.BinaryImageToLabelMap(img)
limg = sitk.LabelMapToLabel(lmap)

shapefilter = sitk.LabelShapeStatisticsImageFilter()
llshape = shapefilter.Execute(limg)

labels = shapefilter.GetLabels()

invalid_seg_labels = []

print "physicals:"
for l in labels:
    if shapefilter.GetPhysicalSize(l) > 1000000.0:
        print l, shapefilter.GetPhysicalSize(l)
    else:
        invalid_seg_labels.append(l)

# print "voxels:"
# for l in labels:
#     if shapefilter.GetNumberOfPixels(l) > 10000.0:
#         print l, shapefilter.GetNumberOfPixels(l)

llfiltered = sitk.GetArrayFromImage(limg)
# indicess = np.where(np.logical_and(llfiltered != 1, llfiltered!=4))
# llfiltered[indicess]=0

for l in invalid_seg_labels:
    llfiltered[llfiltered == l] = 0

newimg = sitk.GetImageFromArray(llfiltered)
newimg.CopyInformation(limg)
sitk.Show(newimg)

sitk.WriteImage(newimg, data + ".kai.mhd")

# 4 3917137.0
# 2578605