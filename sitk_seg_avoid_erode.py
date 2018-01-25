#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:52:22 2018

@author: liuxinglong01
"""

import os
import sys
import math
import SimpleITK as sitk
import numpy as np
from glob import glob

data_path = "/home/liuxinglong01/1HDD/work/vnet.pytorch/orig_imgs/train_subset00"
data_output_path = "/home/liuxinglong01/1HDD/work/vnet.pytorch/orig_imgs/fuzzy_seg"
# data_name = "LKDS-00001.mhd"

LUNG_VOXEL_VAL = 64
BODY_VOXEL_VAL = 192
BED_VOXEL_VAL = 128

def _notice():
    print "*" * 60


def main_fun(data_name):
    img = sitk.ReadImage(os.path.join(data_path, data_name))
    img_w, img_h, img_d = img.GetSize()
    img_phy_w, img_phy_h, img_phy_d = np.array(img.GetSpacing()) * np.array(img.GetSize())
    img_center_x, img_center_y, img_center_z =  np.array(img.GetOrigin()) + \
                                                 np.array([img_phy_w, img_phy_h, img_phy_d])/2.0

    thr_img = sitk.BinaryThreshold(img, -500, 2000)
    dilate_img = sitk.BinaryDilate(thr_img)
    erode_img = sitk.BinaryErode(dilate_img)
    
    all_parts_img = erode_img
    
    lmap = sitk.BinaryImageToLabelMap(erode_img, fullyConnected=True)
    limg = sitk.LabelMapToLabel(lmap)
    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    
    final_iters = 4
    curr_iter = 0
    
    print "looking for largest region"
    
    while len(labels) > 1:
        curr_iter = curr_iter + 1
        print "curr_iter {}, labels {}".format(curr_iter, len(labels)) 
        
        erode_img = sitk.BinaryErode(erode_img)
        lmap = sitk.BinaryImageToLabelMap(erode_img, fullyConnected=True)
        limg = sitk.LabelMapToLabel(lmap)
        shapefilter = sitk.LabelShapeStatisticsImageFilter()
        llshape = shapefilter.Execute(limg)
        labels = shapefilter.GetLabels()
        
        if curr_iter > final_iters:
            print "reduce labels to 1 failed"
            break
    
    print "recovering shape"
    
    dilate_img = sitk.BinaryDilate(erode_img)   
        
    for i in range(curr_iter - 1):
        print "iter {}".format(i)
        dilate_img = sitk.BinaryDilate(dilate_img)

    
    ##########################################################################
    lmap = sitk.BinaryImageToLabelMap(dilate_img , fullyConnected=True)
    limg = sitk.LabelMapToLabel(lmap)
    
    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    body_labels = []
    _notice()
    print "body:"
    largest_size = -1
    largest_size_label = -1 
    for l in labels:
        this_size = shapefilter.GetPhysicalSize(l)
        print l, this_size
        if this_size > 100000.0:
            if this_size > largest_size:
                largest_size = this_size
                largest_size_label = l
    if largest_size_label != -1:
        print "adding {} to body labels".format(largest_size_label)
        body_labels.append(largest_size_label)    
        
    ###########################
    llfiltered = sitk.GetArrayFromImage(limg)
    llmask = np.zeros(llfiltered.shape, dtype=np.uint8)
    if len(body_labels) >= 2:
        op = np.logical_or(llfiltered == body_labels[0], llfiltered == body_labels[1])
        for lidx in range(2, len(body_labels)):
            op = np.logical_or(op, llfiltered == body_labels[lidx])
    elif len(body_labels) == 1:
        op = (llfiltered == body_labels[0])
    else:
        print "body label not found !!" 

    indicess = np.where(op == True)
    llmask[indicess] = BODY_VOXEL_VAL 

    # remove all other tissues
    indicess2 = np.where(llmask == BODY_VOXEL_VAL)
    body_img_array = np.zeros(llmask.shape, dtype=np.uint8)
    body_img_array[indicess2] = 1
    body_img = sitk.GetImageFromArray(body_img_array)
    body_img.CopyInformation(all_parts_img)

    print all_parts_img
    print body_img

    no_body_img = all_parts_img - body_img

    print "operating final labelmapping"
    
    ###########################################################################
    lmap = sitk.BinaryImageToLabelMap(no_body_img, fullyConnected=True)
    limg = sitk.LabelMapToLabel(lmap)
    
    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    
    # seg_labels = []
    bed_labels = []
    _notice()
    print "bed:"
    for l in labels:
        if shapefilter.GetPhysicalSize(l) > 10000.0:
            print l, shapefilter.GetPhysicalSize(l), shapefilter.GetFlatness(l)
            if shapefilter.GetFlatness(l) > 2.0:
                print "adding {} to bed".format(l)
                bed_labels.append(l)
                     
    ###########################
    llfiltered = sitk.GetArrayFromImage(limg)
    op = np.zeros(llfiltered.shape)
    if len(bed_labels) >= 2:
        op = np.logical_or(llfiltered == bed_labels[0], llfiltered == bed_labels[1])
        for lidx in range(2, len(bed_labels)):
            op = np.logical_or(op, llfiltered == bed_labels[lidx])
    elif len(bed_labels)== 1:
        op = (llfiltered == bed_labels[0])
    else:
        print "bed not detected!"

    indicess = np.where(op == True)
    llmask[indicess] = BED_VOXEL_VAL    

    ###########################################################################
    print "anding image"
    # invert to get any other tissues back
    indicess2 = np.where(llmask == BED_VOXEL_VAL)
    bed_img_array = np.zeros(llmask.shape, dtype=np.uint8)
    bed_img_array[indicess2] = 1
    bed_img = sitk.GetImageFromArray(bed_img_array)
    bed_img.CopyInformation(dilate_img)
    
    body_bed_img = body_img + bed_img
    ## 
    invert_img = sitk.InvertIntensity(body_bed_img, maximum=1.0)
    lmap = sitk.BinaryImageToLabelMap(invert_img , fullyConnected=True)
    limg = sitk.LabelMapToLabel(lmap)
    
    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    lung_labels = []
    _notice()
    print "lung: ({},{},{}), ({},{},{}), ({},{},{})".format(img_d, img_w, img_h, \
                                                img_center_x, img_center_y, img_center_z, \
                                                img_phy_d, img_phy_w, img_phy_h)

    for l in labels:
        if shapefilter.GetPhysicalSize(l) > 10000.0:
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
                   lung_labels.append(l)
        
    ###########################
    llfiltered = sitk.GetArrayFromImage(limg)
    op = np.zeros(llfiltered.shape)
    if len(lung_labels) >= 2:
        op = np.logical_or(llfiltered == lung_labels[0], llfiltered == lung_labels[1])
        for lidx in range(2, len(lung_labels)):
            op = np.logical_or(op, llfiltered == lung_labels[lidx])
    elif len(lung_labels) == 1:
        op = (llfiltered == lung_labels[0])
    else:
        print "lung label not found !!"

    indicess = np.where(op == True)
    llmask[indicess] = LUNG_VOXEL_VAL   


    ###########################################################################
    # show
    newimg = sitk.GetImageFromArray(llmask)
    newimg.CopyInformation(limg)
    # sitk.Show(newimg)
    write_path = os.path.join(data_output_path, data_name + "_seg.mhd")
    print "writing to {}".format(write_path)
    sitk.WriteImage(newimg, write_path, True)
    

def process_all(path):
    files = glob(os.path.join(data_path, "*.mhd"))
    
    for file in files:
        print "processing {}".format(file)
        filepath, filename = os.path.split(file)
        main_fun(filename)  


def process_single(filename):
    print "processing {}".format(filename)
    main_fun(filename) 


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        process_single(filename + ".mhd")
    else:
        process_all("")
