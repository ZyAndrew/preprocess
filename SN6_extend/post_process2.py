'''
Author: Shuailin Chen
Created Date: 2021-09-12
Last Modified: 2021-09-16
	content: get the intersection of valid RGB files and masks, and save the valid mask files into .txt file
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
from PIL import Image
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
import colorama
from colorama import Fore, Style
import mylib.file_utils as fu
import mylib.labelme_utils as lu

from tile import (tile_size, tile_slc_path, tile_rgb_path, tile_label_path,
                    tile_mask_path, tile_pauli_path, path, tmp_dir, rgb_prefix, rgb_suffix, PALETTE)



if __name__ == '__main__':

    ''' get the intersection of valid RGB files and masks '''
    print_separate_line(f'get the intersection of valid RGB files and masks')
    splits = os.listdir(tile_mask_path)
    valid_masks = []

    # find valid RGB files
    valid_rgbs = fu.read_file_as_list(osp.join(tile_rgb_path, r'valid.txt'))
    valid_crss = []
    for vrgb in valid_rgbs:
        crs = re.findall(r'\d{6}\_\d{7}', vrgb)
        assert len(crs)==1, f'len of found cfs should be 1, but got {len(crs)}'
        valid_crss.append(crs[0])

    # match mask files to valid RGB files
    for split in splits:
        if osp.isfile(osp.join(tile_mask_path, split)):
            continue

        for mask in os.listdir(osp.join(tile_mask_path, split)):
            mask_crs = re.findall(r'\d{6}\_\d{7}', mask)
            assert len(mask_crs)==1, \
                    f'len of found cfs should be 1, but got {len(crs)}'
            
            if mask_crs[0] in valid_crss:
                valid_masks.append(mask)

        fu.write_file_from_list(valid_masks, osp.join(tile_mask_path, f'valid_{split}.txt'))
        print(f'totally {len(valid_masks)} samples in {split} split')
        valid_masks = []
