'''
Author: Shuailin Chen
Created Date: 2021-09-12
Last Modified: 2021-10-19
	content: get Sinclairv2 (linear normalize on 5% quantile) image of SpaceNet6 dataset. 
    produdure:
        1. 2x2 boxcar filtering
        2. set value = 10*log10(value)
        3. set value<1e-5 = 1e-5
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
from colorama import Fore
import re


def rescale(band, minval, maxval):
    band = 255 * (band - minval) / (maxval - minval)
    band = band.astype("int")
    band[band < 0] = 0
    band[band > 255] = 255
    return band


def get_sinclair_v2(slc_dir: str, dst_dir: str, pauli_dir, orient_file, tmp_dir=None):
    ''' Get Sinclairv2 (linear normalize on 5% quantile) image of SpaceNet6 dataset '''

    print_separate_line(f'generate sinclair v2 begin', Fore.GREEN)
    
    # read orientation infos
    orients = fu.read_file_as_list(orient_file)
    orients = {l.split()[0]: int(l.split()[1]) for l in orients}

    for slc in os.listdir(slc_dir):
        slc_data = tifffile.imread(osp.join(slc_dir, slc))

        # rotate 180 if orientation=1
        timestamp = re.findall(r'2019\d{10}_2019\d{10}', slc)
        assert len(timestamp) == 1
        timestamp = timestamp[0]
        if orients[timestamp]:
            dst_slc_data = slc_data[::-1, ::-1, :]
        else:
            dst_slc_data = slc_data
        
        # extract intensity
        dst_slc_data[np.isnan(dst_slc_data)] = 0

        dst_slc_data = np.square(np.abs(dst_slc_data))

        # 2x2 boxcar filter
        dst_slc_data = cv2.blur(dst_slc_data, (2,2))

        # logarithm
        dst_slc_data = 10*np.log(dst_slc_data)

        # trunct to 1e-5
        dst_slc_data[dst_slc_data<1e-5] = 1e-5
        
        # rescale to [0, 255] per channel
        HH = dst_slc_data[:,:,0]
        VH = dst_slc_data[:,:,2]
        VV = dst_slc_data[:,:,3]

        R = rescale(HH, np.percentile(HH[HH>0], 5), np.percentile(HH[HH>0], 95))
        G = rescale(VV, np.percentile(VV[VV>0], 5), np.percentile(VV[VV>0], 95))
        B = rescale(VH, np.percentile(VH[VH>0], 5), np.percentile(VH[VH>0], 95))

        dst_img = np.dstack((B,G,R))
        dst_img_path = osp.join(dst_dir, slc.replace('tif', 'jpg'))
        iu.save_image_by_cv2(dst_img, dst_img_path, is_bgr=True)
        print(f'writted {dst_img_path}\nPauliRGB: {osp.join(pauli_dir, slc.replace('tif', 'jpg'))}\norient: {orients[timestamp]}')

        if tmp_dir is not None:
            tmp_img_path = osp.join(tmp_dir, 'sinclairv2.jpg')
            iu.save_image_by_cv2(dst_img, tmp_img_path, is_bgr=True)

        print()

    print_separate_line(f'generate sinclair v2 end', Fore.GREEN)


if __name__ == '__main__':
    slc_dir = r'data/SN6_extend/tile_slc/900'
    dst_dir = r'data/SN6_extend/tile_sinclairv2/900'
    pauli_dir = r'data/SN6_extend/tile_pauli/900'
    orient_file = r'data/SN6_extend/SummaryData/SAR_orientations.txt'
    tmp_dir = r'tmp'
    os.makedirs(dst_dir, exist_ok=True)
    get_sinclair_v2(slc_dir, dst_dir, pauli_dir, orient_file=orient_file, tmp_dir=tmp_dir)