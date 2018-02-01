#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:52:22 2018

@author: liuxinglong01
"""

import os
import sys
import math
import time
import logging
import SimpleITK as sitk
import numpy as np
from glob import glob

data_path = "/home/liuxinglong01/1HDD/work/vnet.pytorch/orig_imgs/train_subset00"
data_output_path = "/home/liuxinglong01/1HDD/work/vnet.pytorch/orig_imgs/fuzzy_seg"
# data_name = "LKDS-00001.mhd"

###################################
## labeling constants
LUNG_VOX_VAL = 64
BODY_VOX_VAL = 192
BED_VOX_VAL = 128

###################################
### variables
MIN_LUNG_VOX = 10000.0
MIN_BODY_VOX = 100000.0
MIN_BED_VOX = 5000.0
THRESHOLD = (-500, 2000)

##################################
### debug vars
total_debug_time = 0.0

###################################
### global medical image gpu filter
def check_medgpu():
    try:
        medImageFilter = sitk.MedImageGPUFilter()
        _notice()
        logging.debug("setting med image filter to gpu")
    except:
        logging.warning("no GPU impl for med image filter found, revert to CPU version")
        medImageFilter = sitk
    return medImageFilter


def _notice():
    logging.debug("*" * 60)


def init_logging(logFilename):
    """
    Init for logging
    """
    logging.basicConfig(
        level    = logging.DEBUG,
        format   = 'LINE %(lineno)-4d  %(levelname)-8s %(message)s',
        datefmt  = '%m-%d %H:%M',
        filename = logFilename,
        filemode = 'w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main_fun(data_name):
    global total_debug_time

    medImageFilter = check_medgpu()

    img = sitk.ReadImage(os.path.join(data_path, data_name))
    img_w, img_h, img_d = img.GetSize()
    img_phy_w, img_phy_h, img_phy_d = np.array(img.GetSpacing()) * np.array(img.GetSize())
    img_center_x, img_center_y, img_center_z =  np.array(img.GetOrigin()) + \
                                                 np.array([img_phy_w, img_phy_h, img_phy_d])/2.0
    
    ###########################
    # debugging time
    start_time = time.time()
    ###########################

    thr_img = sitk.BinaryThreshold(img, -500, 2000)
    # thr_img = sitk.BinaryFillhole(thr_img)

    dilate_img = medImageFilter.BinaryDilate(thr_img)
    start_img = medImageFilter.BinaryErode(dilate_img)
    
    ###########################
    # sitk.Show(start_img)
    ###########################

    ###########################################################################
    # invert to get any other tissues back
    invert_img = sitk.InvertIntensity(start_img, maximum=1.0)
    lmap = sitk.BinaryImageToLabelMap(invert_img)
    limg = sitk.LabelMapToLabel(lmap)
    
    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    lung_labels = []
    _notice()
    logging.debug("lung: ({},{},{}), ({},{},{}), ({},{},{})".format(img_d, img_w, img_h, \
                                                img_center_x, img_center_y, img_center_z, \
                                                img_phy_d, img_phy_w, img_phy_h))

    for l in labels:
        if shapefilter.GetPhysicalSize(l) > MIN_LUNG_VOX:
            bbox = shapefilter.GetBoundingBox(l)
            # at the center of the image
            # size can not occupy the whole image
            region_d = abs(bbox[5] - bbox[2]) 
            region_w = abs(bbox[4] - bbox[1]) 
            region_h = abs(bbox[3] - bbox[0]) 
            center_x, center_y, center_z = shapefilter.GetCentroid(l)

            logging.debug("{},{}, ({},{},{}), ({},{},{})".\
                format(l, shapefilter.GetPhysicalSize(l), region_d, region_w, region_h, center_x, center_y, center_z))

            # NOTE, we do NOT setence image depth here! since lung sometimes would 
            # occupy the whole image alongside the depth direction
            # if  region_d < 0.9 * img_d and \
            # 大小小于90%原图大小，且中心在原图中心1/4以内
            if  region_w < 0.9 * img_w and \
                region_h < 0.9 * img_h and \
                abs(center_x - img_center_x) < img_phy_w / 4 and \
                abs(center_y - img_center_y) < img_phy_h / 4 and \
                abs(center_z - img_center_z) < img_phy_d / 4:
                   logging.debug("adding {} to lung labels".format(l))
                   lung_labels.append(l)

    ###########################
    llfiltered = sitk.GetArrayFromImage(limg)
    llmask = np.zeros(llfiltered.shape)
    op = np.zeros(llfiltered.shape)
    if len(lung_labels) >= 2:
        op = np.logical_or(llfiltered == lung_labels[0], llfiltered == lung_labels[1])
        for lidx in range(2, len(lung_labels)):
            op = np.logical_or(op, llfiltered == lung_labels[lidx])
    elif len(lung_labels) == 1:
        op = (llfiltered == lung_labels[0])
    else:
        logging.debug("lung label not found !!")

    lung_indicess = np.where(op == True)
    # llmask[lung_indicess] = LUNG_VOX_VAL   

    lung_img_arr = np.zeros(llfiltered.shape, dtype=np.uint8)
    lung_img_arr[lung_indicess] = 1
    lung_img = sitk.GetImageFromArray(lung_img_arr)
    lung_img.CopyInformation(start_img)
    erode_img = start_img + lung_img
    
    ######################
    # sitk.Show(erode_img)
    sitk.WriteImage(erode_img, os.path.join(data_output_path, data_name + "_whole.mhd"), True)
    ######################

    ##########################################################################
    lmap = sitk.BinaryImageToLabelMap(erode_img)
    limg = sitk.LabelMapToLabel(lmap)
    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    
    final_iters = 5
    curr_iter = 0
    
    logging.debug("looking for largest region")
    
    while len(labels) > 20:
        curr_iter = curr_iter + 1
        logging.debug("curr_iter {}, labels {}".format(curr_iter, len(labels))) 
        
        erode_img = medImageFilter.BinaryErode(erode_img)
        lmap = sitk.BinaryImageToLabelMap(erode_img)
        limg = sitk.LabelMapToLabel(lmap)
        shapefilter = sitk.LabelShapeStatisticsImageFilter()
        llshape = shapefilter.Execute(limg)
        labels = shapefilter.GetLabels()
        
        if curr_iter > final_iters:
            logging.debug("reduce labels to 1 failed")
            break
    
    logging.debug("recovering shape iter {0}".format(1))
    dilate_img = medImageFilter.BinaryDilate(erode_img)   
        
    for i in range(1, curr_iter):
        logging.debug("recovering shape iter {0}".format(i))
        dilate_img = medImageFilter.BinaryDilate(dilate_img)

    # dilate_img = dilate_img - lung_img
    ##########################################################################
    lmap = sitk.BinaryImageToLabelMap(dilate_img)
    limg = sitk.LabelMapToLabel(lmap)
    
    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    body_labels = []
    _notice()
    logging.debug("body:")
    largest_size = -1
    largest_size_label = -1 
    for l in labels:
        this_size = shapefilter.GetPhysicalSize(l)
        # 滤除较小体素数量的
        if this_size > MIN_BODY_VOX:
            logging.debug("{}, {}".format(l, this_size))
            if this_size > largest_size:
                largest_size = this_size
                largest_size_label = l
    if largest_size_label != -1:
        logging.debug("adding {} to body labels".format(largest_size_label))
        body_labels.append(largest_size_label)    
        
    ###########################
    llfiltered = sitk.GetArrayFromImage(limg)
    if len(body_labels) >= 2:
        op = np.logical_or(llfiltered == body_labels[0], llfiltered == body_labels[1])
        for lidx in range(2, len(body_labels)):
            op = np.logical_or(op, llfiltered == body_labels[lidx])
    elif len(body_labels) == 1:
        op = (llfiltered == body_labels[0])
    else:
        logging.debug("body label not found !!")

    body_indicess = np.where(op == True)
    # llmask[body_indicess] = BODY_VOX_VAL 

    # remove all other tissues
    whole_body_img_array = np.zeros(llmask.shape, dtype=np.uint8)
    whole_body_img_array[body_indicess] = 1
    whole_body_img = sitk.GetImageFromArray(whole_body_img_array)
    whole_body_img.CopyInformation(start_img)
    body_img = whole_body_img - lung_img

    ######################
    # sitk.Show(whole_body_img)
    ######################
        
    ###########################################################################
    no_body_no_lung_img = start_img - body_img - lung_img

    dil_no_body_no_lung_img = medImageFilter.BinaryDilate(no_body_no_lung_img)
    lmap = sitk.BinaryImageToLabelMap(dil_no_body_no_lung_img)
    limg = sitk.LabelMapToLabel(lmap)

    shapefilter = sitk.LabelShapeStatisticsImageFilter()
    llshape = shapefilter.Execute(limg)
    labels = shapefilter.GetLabels()
    
    # seg_labels = []
    bed_labels = []
    _notice()
    logging.debug("bed:")
    for l in labels:
        if shapefilter.GetPhysicalSize(l) > MIN_BED_VOX:
            logging.debug("{},{},{}".format(l, shapefilter.GetPhysicalSize(l), shapefilter.GetFlatness(l)))
            center_x, center_y, center_z = shapefilter.GetCentroid(l)
            # 图像上下1/4区间
            if shapefilter.GetFlatness(l) > 2.0 and \
                abs(center_y - img_center_y) > img_phy_h / 4:
                logging.debug("adding {} to bed".format(l))
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
        logging.debug("bed not detected!")

    bed_indicess = np.where(op == True)
    # llmask[bed_indicess] = BED_VOX_VAL    

    bed_img_array = np.zeros(llmask.shape, dtype=np.uint8)
    bed_img_array[bed_indicess] = 1
    bed_img = sitk.GetImageFromArray(bed_img_array)
    bed_img.CopyInformation(dilate_img)

    # further post-processing
    bed_img = medImageFilter.BinaryErode(bed_img)
    bed_img_array = sitk.GetArrayFromImage(bed_img)
    bed_indicess = np.where(bed_img_array == 1)   

    ######################
    # sitk.Show(bed_img)
    ######################

    ###########################
    # debugging time
    end_time = time.time()
    this_time = end_time - start_time
    total_debug_time = total_debug_time + this_time
    logging.debug( "data {0}, process time {1}".format(data_name, this_time) )
    ###########################

    ###########################################################################
    # show
    # NOTE body indices include lung indices, so set lung last
    llmask[bed_indicess] = BED_VOX_VAL    
    llmask[body_indicess] = BODY_VOX_VAL 
    llmask[lung_indicess] = LUNG_VOX_VAL   
    
    newimg = sitk.GetImageFromArray(llmask)
    newimg.CopyInformation(limg)
    # # sitk.Show(newimg)
    write_path = os.path.join(data_output_path, data_name + "_seg.mhd")
    logging.debug("writing to {}".format(write_path))
    sitk.WriteImage(newimg, write_path, True)
    

def process_all(path):
    files = glob(os.path.join(data_path, "*.mhd"))
    
    global total_debug_time

    for file in files:
        _notice()
        _notice()
        
        logging.debug("processing {}".format(file))
        filepath, filename = os.path.split(file)
        main_fun(filename)  

    logging.debug( "total time {0}, average time {1}".format(total_debug_time, total_debug_time / len(files)) )

def process_single(filename):
    _notice()
    _notice()
    
    logging.debug("processing {}".format(filename))
    main_fun(filename) 


if __name__ == "__main__":
    init_logging("./sitk_seg.log")

    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        process_single(filename + ".mhd")
    else:
        process_all("")
