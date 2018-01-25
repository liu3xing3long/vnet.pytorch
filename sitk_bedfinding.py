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
LKDS-00016_Threshold.mha'

img = sitk.ReadImage(data)
img_w, img_h, img_d = img.GetSize()
img_phy_w, img_phy_h, img_phy_d = np.array(img.GetSpacing()) * np.array(img.GetSize())
img_center_x, img_center_y, img_center_z =  np.array(img.GetOrigin()) + \
                                                 np.array([img_phy_w, img_phy_h, img_phy_d])/2.0

lmap = sitk.BinaryImageToLabelMap(img, fullyConnected=True)
limg = sitk.LabelMapToLabel(lmap)

shapefilter = sitk.LabelShapeStatisticsImageFilter()
llshape = shapefilter.Execute(limg)

labels = shapefilter.GetLabels()

seg_labels = []
bed_labels = []
print "physicals:"
for l in labels:
    if shapefilter.GetPhysicalSize(l) > 10000.0:
        print l, shapefilter.GetPhysicalSize(l), shapefilter.GetFlatness(l)
        bbox = shapefilter.GetBoundingBox(l)
        # at the center of the image
        # size can not occupy the whole image
        region_d = abs(bbox[5] - bbox[2]) 
        region_w = abs(bbox[4] - bbox[1]) 
        region_h = abs(bbox[3] - bbox[0]) 
        center_x, center_y, center_z = shapefilter.GetCentroid(l)

        print "{},({},{},{}), ({},{},{})".\
            format(l, region_d, region_w, region_h, center_x, center_y, center_z)

        # NOTE, we do NOT setence image depth here! since lung sometimes would 
        # occupy the whole image alongside the depth direction
        # if  region_d < 0.9 * img_d and \
        if  region_w < 0.9 * img_w and \
            region_h < 0.9 * img_h and \
            abs(center_x - img_center_x) < img_phy_w / 4 and \
            abs(center_y - img_center_y) < img_phy_h / 4 and \
            abs(center_z - img_center_z) < img_phy_d / 4:
               print "adding {} to lung labels".format(l)
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

