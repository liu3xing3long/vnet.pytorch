#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:26:40 2017

@author: liuxinglong01
"""

import shutil
import random
import os

source = "luna16_1mm"
target = "luna16_1mm_partial_small"

dir1 = "normalized_ct_images"
dir2 = "normalized_lung_masks"

fp = open("list", "r")

dat = []
while True:
    tmp = fp.readline()
    if not tmp:
        break

    tmp = tmp.strip('\n')
    dat.append(tmp)

fp.close()


nData = len(dat)
nPartial = 4

partial_idx = random.sample(range(nData), nPartial)

for idx in partial_idx:
    shutil.copy(os.path.join(source, dir1, dat[idx] + ".mhd"),
                os.path.join(target, dir1, dat[idx] + ".mhd"))

    shutil.copy(os.path.join(source, dir1, dat[idx] + ".raw"), 
                os.path.join(target, dir1, dat[idx] + ".raw"))
    
    shutil.copy(os.path.join(source, dir2, dat[idx] + ".mhd"), 
                os.path.join(target, dir2, dat[idx] + ".mhd"))

    shutil.copy(os.path.join(source, dir2, dat[idx] + ".raw"), 
                os.path.join(target, dir2, dat[idx] + ".raw"))