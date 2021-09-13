'''
Author: Shuailin Chen
Created Date: 2021-09-12
Last Modified: 2021-09-13
	content: post-processing, including:
        1. generate PauliRGB, 
        2. remove invalid slc files
        3. remove labels beyond boundary
'''

import os
import os.path as osp
import numpy as np
import cv2
from glob import glob
import mylib.polSAR_utils as psr
import mylib.image_utils as iu
from mylib.types import print_separate_line
import mylib.labelme_utils as lu
import re
import tifffile
import shutil

import solaris as sol
import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.label as label
import solaris.preproc.image as image
import solaris.preproc.sar as sar
from solaris.data import data_dir
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union


''' existing path '''
path = r'/home/csl/code/preprocess/data/SN6_extend'
tmp_dir = r'/home/csl/code/preprocess/tmp'

''' dirs of output files '''
tile_size = 900
tile_slc_path = osp.join(path, r'tile_slc', str(tile_size))
tile_rgb_path = osp.join(path, r'tile_rgb', str(tile_size))
tile_label_path = osp.join(path, r'tile_label', str(tile_size))
tile_mask_path = osp.join(path, r'tile_mask', str(tile_size))
tile_pauli_path = osp.join(path, r'tile_pauli', str(tile_size))

''' generate pauli rgb from slc data, remove invali data '''
print_separate_line(f'generating PauliRGB')
slc_suffix = r'tif'
pauli_suffix = r'jpg'
slcs = glob(osp.join(tile_slc_path, r'slc_*.tif'))
print(f'{len(slcs)} samples in total')
for slc in slcs:
    slc_data = tifffile.imread(slc)
    pauli = psr.rgb_by_s2(slc_data.transpose(2, 0, 1), if_mask=True, is_print=False)
    if pauli is None:
        ''' all pixels are black, invalid sample'''
        os.remove(slc)
        continue
    pauli_path = slc.replace(tile_slc_path, tile_pauli_path) \
                    .replace(slc_suffix, pauli_suffix)
    print(f'saving {pauli_path}')
    iu.save_image_by_cv2(pauli, pauli_path, is_bgr=False)

''' remove remove labels beyond boundary '''
print_separate_line(f'removing invalid labels')