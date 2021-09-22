'''
Author: Shuailin Chen
Created Date: 2021-09-12
Last Modified: 2021-09-17
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
                    tile_mask_path, tile_pauli_path, path, tmp_dir, rgb_prefix, rgb_suffix, PALETTE, mask_prefix, mask_suffix, slc_suffix, pauli_suffix)


if __name__ == '__main__':

    # ''' select the valid RGB samples through black regions '''
    # print_separate_line(f'removing invalid labels')
    # rgbs = glob(osp.join(tile_rgb_path, f'SN6_*.tif'))
    # print(f'{len(rgbs)} samples in total')
    # valid_samples = []
    # for rgb_path in rgbs:
    #     rgb_img = Image.open(rgb_path)
    #     mask_path = rgb_path.replace(tile_rgb_path, tile_mask_path) \
    #                         .replace(rgb_prefix, mask_prefix) \
    #                         .replace(rgb_suffix, mask_suffix)
    #     # gif_path = mask_path.replace('png', 'gif')
    #     # mask_img = Image.open(mask_path)
    #     bimasked = rgb_img.convert('L')
    #     bimasked = Image.fromarray(np.asarray(bimasked)>0)

    #     # bimasked = np.asarray(bimasked).astype(np.uint8)
    #     kernel_size = 19
    #     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #     closed = cv2.morphologyEx(bimasked, cv2.MORPH_CLOSE, kernel)

    #     row_sum = closed.sum(axis=0)
    #     col_sum = closed.sum(axis=1)
    #     assert len(np.unique(row_sum))<3 and (len(np.unique(row_sum)))<3, \
    #             f'unique values of row sum and column sum should <=2 ,but got row sum={row_sum}, column sum={col_sum}'
    #     values = np.unique(closed)
    #     if len(values)==1:
    #         if values == 0:
    #             ''' all invalid '''
    #             print(f'{Fore.RED}all pixels of {rgb_path} are invalid')
    #             continue
    #             # os.remove(rgb_path)
    #             # os.remove(mask_path)
    #         elif values == 1:
    #             ''' all valid '''
    #             print(f'{Fore.RESET}all pixels of {rgb_path} are valid')
    #             valid_samples.append(rgb_path)
    #     elif len(values) == 2:
    #         binc = np.bincount(closed.flatten())
    #         if binc[1] / binc[0] < 0.005:
    #             ''' valid region is too small, discard '''
    #             print(f'{Fore.RED}valid region of {rgb_path} is too small\nits mask {mask_path}')
    #             continue
    #             # os.remove(rgb_path)
    #             # os.remove(mask_path)
    #             # if osp.isfile(gif_path):
    #             #     os.remove(gif_path)
    #         else:
    #             # print(f'{Fore.GREEN}pixels of {rgb_path} are partly valid')
    #             # mask_img *= closed
    #             # print(f'{Fore.GREEN}writting {mask_path}')
    #             # print(f'{Fore.GREEN}writting {gif_path}')
    #             # lu.lblsave(mask_path, mask_img, colormap=PALETTE)
    #             # rgb_img.save(gif_path, format='GIF',
    #             #             append_images=[Image.fromarray(np.tile(closed[..., None], (1, 1, 3))*255)],
    #             #             loop=0, save_all=True, duration=700)
    #             valid_samples.append(rgb_path)
    #     else:
    #         raise ValueError(f'should not exist values > 3')

    # fu.write_file_from_list(valid_samples,
    #                             osp.join(tile_rgb_path, 'valid.txt'))


    ''' generate pauli rgb from slc data, remove invalid data '''
    # find crs of valid masks
    print_separate_line(f'select valid PauliRGB')
    valid_mask_crss = dict()
    for split in os.listdir(tile_mask_path):
        filepath = osp.join(tile_mask_path, split)
        if osp.isfile(filepath):
            valid_mask = fu.read_file_as_list(filepath)
            tmp = []
            for vmk in valid_mask:
                crs = re.findall(r'\d{6}\_\d{7}', vmk)
                assert len(crs)==1, f'len of found cfs should be 1, but got {len(crs)}'
                tmp.append(crs[0])
            valid_mask_crss.update({split: tmp})

    # match valid SLCs to valid masks
    valid_slcs = dict()
    slcs = glob(osp.join(tile_slc_path, r'slc_*.tif'))
    print(f'{len(slcs)} slc samples in total')
    for slc in slcs:
        # remove SLC image with all black region
        print(f'reading {slc}')
        slc_data = tifffile.imread(slc)
        pauli = psr.rgb_by_s2(slc_data.transpose(2, 0, 1), if_mask=True, is_print=False)

        if pauli is None:
            ''' all pixels are black, invalid sample'''
            print(f'{Fore.RED}all of entries of {slc} is nan, removing{Fore.RESET}')
            os.remove(slc)
            continue
        
        pauli_path = slc.replace(tile_slc_path, tile_pauli_path) \
                        .replace(slc_suffix, pauli_suffix)
        iu.save_image_by_cv2(pauli, pauli_path, is_bgr=False)
        print(f'saving {pauli_path}')

        bi_pauli = cv2.cvtColor(pauli, cv2.COLOR_RGB2GRAY) > 0
        bi_pauli = bi_pauli.astype(np.uint8)
        kernel_size = 19
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(bi_pauli, cv2.MORPH_CLOSE, kernel)

        values = np.unique(closed)
        if len(values)==1:
            if values == 0:
                ''' all invalid '''
                print(f'{Fore.RED} all pixels of {pauli_path} are invalid,\
                \nslc: {slc}, removing{Fore.RESET}')
                os.remove(pauli_path)
                os.remove(slc)
                continue
            elif values == 1:
                ''' all valid '''
                print(f'{Fore.RESET}all pixels of {pauli_path} are valid')
        elif len(values) == 2:
            binc = np.bincount(closed.flatten())
            if binc[1] / binc[0] < 0.1:
                ''' valid region is too small, discard '''
                print(f'{Fore.RED}valid region of {pauli_path} is too small,\
                \nslc: {slc}, removing')
                os.remove(pauli_path)
                os.remove(slc)
                continue
            else:
                print(f'{Fore.GREEN}pixels of {pauli_path} are partly valid')
        else:
            raise ValueError(f'should not exist values > 3')


        # find slc files' split
        slc_crs = re.findall(r'(\d{6}\_\d{7})\.tif', slc)
        assert len(slc_crs)==1, \
                f'len of found slc_crs should be 1, but got {len(slc_crs)}'

        for split, masks in valid_mask_crss.items():
            base_slc = osp.basename(slc)
            if slc_crs[0] in masks:
                print(f'{Fore.YELLOW}add {base_slc} to {split}{Fore.RESET}')
                if valid_slcs.get(split, None):
                    valid_slcs.get(split).append(base_slc)
                else:
                    valid_slcs[split] = [base_slc]
                break

    # write to .txt files
    for split, slcs in valid_slcs.items():
        fu.write_file_from_list(slcs, osp.join(tile_slc_path, split))
