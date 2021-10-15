'''
Author: Shuailin Chen
Created Date: 2021-09-12
Last Modified: 2021-10-15
	content: get Sinclairv2 (linear normalize on 5% quantile) image of SpaceNet6 dataset
'''

import os
import os.path as osp
import numpy as np
import cv2
from glob import glob
import tifffile
import mylib.polSAR_utils as psr
import mylib.image_utils as iu
import mylib.file_utils as fu
import mylib.labelme_utils as lu
import mylib.mathlib as ml
from mylib.types import print_separate_line
from colorama import Fore, Style


def rescale(band, minval, maxval):
    band = 255 * (band - minval) / (maxval - minval)
    band = band.astype("int")
    band[band < 0] = 0
    band[band > 255] = 255
    return band


def get_sinclair_v2(slc_dir: str, dst_dir: str):
    ''' Get Sinclairv2 (linear normalize on 5% quantile) image of SpaceNet6 dataset '''

    print_separate_line(f'generate sinclair v2 begin', Fore.GREEN)
    for slc in os.listdir(slc_dir):
        slc_data = tifffile.imread(osp.join(slc_dir, slc))
        
        HH = slc_data[:,:,0]
        VH = slc_data[:,:,2]
        VV = slc_data[:,:,3]

        HH[np.isnan(HH)] = 0
        VH[np.isnan(VH)] = 0
        VV[np.isnan(VV)] = 0

        HH = np.square(np.abs(HH))
        VH = np.square(np.abs(VH))
        VV = np.square(np.abs(VV))

        R = rescale(HH, np.percentile(HH[HH>0], 5), np.percentile(HH[HH>0], 95))
        G = rescale(VV, np.percentile(VV[VV>0], 5), np.percentile(VV[VV>0], 95))
        B = rescale(VH, np.percentile(VH[VH>0], 5), np.percentile(VH[VH>0], 95))
        out_img = np.dstack((B,G,R))

        dst_img_path = osp.join(dst_dir, slc.replace('tif', 'jpg'))
        cv2.imwrite(dst_img_path, out_img)
        print(f'writted {dst_img_path}')
        print()

    print_separate_line(f'generate sinclair v2 end', Fore.GREEN)





if __name__ == '__main__':
    slc_dir = r'data/SN6_extend/tile_slc/900'
    dst_dir = r'data/SN6_extend/tile_sinclairv2/900'
    os.makedirs(dst_dir, exist_ok=True)
    get_sinclair_v2(slc_dir, dst_dir)