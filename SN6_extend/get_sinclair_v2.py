'''
Author: Shuailin Chen
Created Date: 2021-09-12
Last Modified: 2021-10-19
	content: get Sinclairv2 (linear normalize on 5% quantile) image of SpaceNet6 dataset. 
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
from mylib.types import print_separate_line, re_find_only_one
from colorama import Fore
import re
import shutil
import warnings

from merge_duplicate_label import read_label_png


def rescale(band, minval, maxval):
    band = 255 * (band - minval) / (maxval - minval)
    band = band.astype("int")
    band[band < 0] = 0
    band[band > 255] = 255
    return band


def get_sinclair_v2(slc_dir: str, dst_dir: str, pauli_dir, dst_mask_dir, mask_dir, orient_file, tmp_dir=None):
    ''' Get Sinclairv2 image (format used in SpaceNet 6 official) of SpaceNet6 dataset. Its RGB are HH, VV, VH
    
    NOTE: I can not reproduce the results of provide by the author, so I use my method, to get similar results. Generally, my results are a bit brighter than the offcial
    '''

    print_separate_line(f'generate sinclair v2 begin', Fore.GREEN)
    
    # read orientation infos
    orients = fu.read_file_as_list(orient_file)
    orients = {l.split()[0]: int(l.split()[1]) for l in orients}

    # read mask split
    mask_train_split = fu.read_file_as_list(osp.join(mask_dir, f'valid_train.txt'))
    mask_val_split = fu.read_file_as_list(osp.join(mask_dir, f'valid_val.txt'))
    os.makedirs(osp.join(dst_mask_dir, 'train'), exist_ok=True)
    os.makedirs(osp.join(dst_mask_dir, 'val'), exist_ok=True)
    shutil.copy(osp.join(mask_dir, f'valid_train.txt'),
                osp.join(dst_mask_dir, f'valid_train.txt'))
    shutil.copy(osp.join(mask_dir, f'valid_val.txt'),
                osp.join(dst_mask_dir, f'valid_val.txt'))

    for ii, slc in enumerate(os.listdir(slc_dir)):
        if not slc.endswith('.tif'):
            continue
        slc_data = tifffile.imread(osp.join(slc_dir, slc))
        crs = re_find_only_one(r'(\d{6}_\d{7})\.tif', slc)

        mask_filename = f'geoms_{crs}.png'
        if mask_filename in mask_train_split:
            mask_data = read_label_png(osp.join(mask_dir, 'train', mask_filename))
            mask_split = 'train'
            # mask_filename = osp.join('g')
        elif mask_filename in mask_val_split:
            mask_data = read_label_png(osp.join(mask_dir, 'val', mask_filename))
            mask_split = 'val'
        else:
            warnings.warn(f'can not find corresponding mask file')

        # rotate 180 if orientation=1
        timestamp = re.findall(r'2019\d{10}_2019\d{10}', slc)
        assert len(timestamp) == 1
        timestamp = timestamp[0]
        if orients[timestamp]:
            dst_slc_data = slc_data[::-1, ::-1, :]
            mask_data = mask_data[::-1, ::-1]
            # dst_slc_data = slc_data
        else:
            dst_slc_data = slc_data
        
        # get sinclair v2, add 2x2 boxcar filtering, according to the SpaceNet6 paper
        sinclairv2 = psr.rgb_by_s2(dst_slc_data.transpose(2, 0, 1), type='sinclairv2', if_mask=True, is_print=False, boxcar_ksize=2)

        # extract intensity
        dst_slc_data[np.isnan(dst_slc_data)] = 0

        dst_slc_data = np.square(np.abs(dst_slc_data))

        # # 2x2 boxcar filter
        # dst_slc_data = cv2.blur(dst_slc_data, (2,2))

        # # logarithm
        # dst_slc_data = 10*np.log10(dst_slc_data)

        # # trunct to 1e-5
        # dst_slc_data[np.logical_and(-150<dst_slc_data, dst_slc_data<1e-5)] = 1e-5
        # dst_slc_data[dst_slc_data<0] = 0
        
        # # rescale to [0, 255] per channel
        # HH = dst_slc_data[:, :, 0]
        # VH = dst_slc_data[:, :, 2]
        # VV = dst_slc_data[:, :, 3]

        # R = rescale(HH, np.percentile(HH[HH>0], 1), np.percentile(HH[HH>0], 99))
        # G = rescale(VV, np.percentile(VV[VV>0], 1), np.percentile(VV[VV>0], 99))
        # B = rescale(VH, np.percentile(VH[VH>0], 1), np.percentile(VH[VH>0], 99))

        # dst_img = np.dstack((B,G,R))
        dst_img_path = osp.join(dst_dir, slc.replace('tif', 'jpg'))
        iu.save_image_by_cv2(sinclairv2, dst_img_path, is_bgr=False)
        dst_mask_path = osp.join(dst_mask_dir, mask_split, mask_filename)
        lu.lblsave(dst_mask_path, mask_data,
                    colormap=np.array([[0, 0, 0], [255, 255, 255]]))
        print(f'{ii}:\nwritted {dst_img_path}\nPauliRGB: {osp.join(pauli_dir, slc.replace("tif", "jpg"))}\norient: {orients[timestamp]}\nmax intensity: {dst_slc_data.max()}')

        if tmp_dir is not None:
            tmp_img_path = osp.join(tmp_dir, 'sinclairv2.jpg')
            tmp_pauli_path = osp.join(tmp_dir, 'pauli.jpg')
            # iu.save_image_by_cv2(dst_img, tmp_img_path, is_bgr=True)
            # iu.save_image_by_cv2(pauli, tmp_pauli_path, is_bgr=False)
            iu.save_as_gif([sinclairv2, mask_data*255], osp.join(tmp_dir, 'rotated.gif'))
            print(f'\nsinclairv2: {tmp_img_path}\npauli: {tmp_pauli_path}')

        print()

    print_separate_line(f'generate sinclair v2 end', Fore.GREEN)


def rotate_masks(src_dir, dst_dir, orient_file):
    ''' Rotate masks '''

    orients = fu.read_file_as_list(orient_file)
    orients = {l.split()[0]: int(l.split()[1]) for l in orients}

    
    splits = ('train', 'val')
    for sp in splits:
        split_file = fu.read_file_as_list(osp.join(src_dir,
                                                f'valid_{sp}.txt'))
        for file in split_file:
            mask = read_label_png(osp.join(src_dir, sp, file))

            # rotate 180 if orientation=1
            timestamp = re.findall(r'2019\d{10}_2019\d{10}', file)
            assert len(timestamp) == 1
            timestamp = timestamp[0]
            if orients[timestamp]:
                dst_slc_data = slc_data[::-1, ::-1, :]
                # dst_slc_data = slc_data
            else:
                dst_slc_data = slc_data
            


if __name__ == '__main__':
    slc_dir = r'data/SN6_extend/tile_slc/900'
    dst_dir = r'data/SN6_extend/tile_sinclairv2_rotated/900'
    mask_dir = r'data/SN6_extend/tile_mask/900'
    dst_mask_dir = r'data/SN6_extend/tile_mask_rotated/900'
    pauli_dir = r'data/SN6_extend/tile_pauli/900'
    orient_file = r'data/SN6_extend/SummaryData/SAR_orientations.txt'
    # tmp_dir = r'tmp'
    tmp_dir = None
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    get_sinclair_v2(slc_dir, dst_dir, pauli_dir, mask_dir=mask_dir, orient_file=orient_file, tmp_dir=tmp_dir, dst_mask_dir=dst_mask_dir)