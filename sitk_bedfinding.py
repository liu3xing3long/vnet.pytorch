#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:38:25 2018

@author: liuxinglong01
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:19:21 2018

@author: liuxinglong01
"""

import numpy as np
import SimpleITK as sitk


# data = '/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/results/LKDS-00065.mhd'
data = '/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/test/\
LKDS-00013_Threshold.mha'

img = sitk.ReadImage(data)

lmap = sitk.BinaryImageToLabelMap(img, fullyConnected=True)
limg = sitk.LabelMapToLabel(lmap)

shapefilter = sitk.LabelShapeStatisticsImageFilter()
llshape = shapefilter.Execute(limg)

labels = shapefilter.GetLabels()

seg_labels = []
bed_labels = []
print "physicals:"
for l in labels:
    if shapefilter.GetPhysicalSize(l) > 1000000.0:
        print l, shapefilter.GetPhysicalSize(l), shapefilter.GetFlatness(l)
        seg_labels.append(l)
        #if shapefilter.GetFlatness(l) > 3.0:
        #   seg_labels.append(l)
                

llfiltered = sitk.GetArrayFromImage(limg)
llmask = np.zeros(llfiltered.shape)

if len(seg_labels) >= 2:
    op = np.logical_or(llfiltered == seg_labels[0], llfiltered == seg_labels[1])
    for lidx in range(2, len(seg_labels)):
        op = np.logical_or(op, llfiltered == seg_labels[lidx])
else:
    op = (llfiltered == seg_labels[0])
       
indicess = np.where(op == True)
llmask[indicess] = 1    
    
llfiltered = llfiltered * llmask

llfiltered[llfiltered==14] = 64
llfiltered[llfiltered==356] = 128
llfiltered[llfiltered==414] = 190

newimg = sitk.GetImageFromArray(llfiltered)
newimg.CopyInformation(limg)
sitk.Show(newimg)

