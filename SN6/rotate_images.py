'''
Author: Shuailin Chen
Created Date: 2021-10-16
Last Modified: 2021-10-18
	content: rotate image and mask when orientation=1
'''

import os
import os.path as osp
import tifffile
import cv2
import numpy as np
import mylib.file_utils as fu
import mylib.labelme_utils as lu
import re
from PIL import Image

from unsup_seg import read_label_png


def rotate_img_labels(img_src_dir,
                img_dst_dir,
                mask_src_dir,
                mask_dst_dir,
                orient_file,
                colormap=None,
                tmp_dir=None):
    ''' Rotate images and mask according to its direction '''

    # preparation
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(mask_dst_dir, exist_ok=True)

    if colormap is None:
        colormap = np.array([[128, 128, 128], [0, 0, 0], [255, 255, 255]])

    # read orientation infos
    orients = fu.read_file_as_list(orient_file)
    orients = {l.split()[0]: int(l.split()[1]) for l in orients}

    # rotate image
    for img_name in os.listdir(img_src_dir):
        img = tifffile.imread(osp.join(img_src_dir, img_name))
        mask_name = img_name.replace('tif', 'png').replace('SAR-Intensity', 'PS-RGB')
        mask = read_label_png(osp.join(mask_src_dir, mask_name),
                                check_path=False)
        
        timestamp = re.findall(r'2019\d{10}_2019\d{10}', img_name)
        assert len(timestamp) == 1
        timestamp = timestamp[0]

        if orients[timestamp]:
            dst_img = img[::-1, ::-1, :]
            dst_mask = mask[::-1, ::-1]
        else:
            dst_img = img
            dst_mask = mask

        # save images
        img_dst_path = osp.join(img_dst_dir, img_name)
        mask_dst_path = osp.join(mask_dst_dir, mask_name)
        tifffile.imwrite(img_dst_path, dst_img)
        lu.lblsave(mask_dst_path, dst_mask, colormap=colormap)
        
        print(f'src img: {osp.join(img_src_dir, img_name)}\nsrc label: {osp.join(mask_src_dir, mask_name)}\ndst img:{img_dst_path}\ndst mask: {mask_dst_path}\norient: {orients[timestamp]}')
        print()

        # save images
        if tmp_dir is not None:
            gif_ori_path = osp.join(tmp_dir, 'ori.gif')
            gif_dst_path = osp.join(tmp_dir, 'dst.gif')
            img = Image.fromarray(img)
            mask = Image.fromarray(mask*128)
            dst_img = Image.fromarray(dst_img)
            dst_mask = Image.fromarray(dst_mask*128)
            img.save(gif_ori_path, format='GIF', append_images=[mask], loop=0, save_all=True, duration=700)
            dst_img.save(gif_dst_path, format='GIF', append_images=[dst_mask], loop=0, save_all=True, duration=700)
            print(f'ori gif: {gif_ori_path}\ndst gif: {gif_dst_path}')
            print()


def rotate_imgs(img_src_dir,
                img_dst_dir,
                orient_file,
                tmp_dir=None,
                colormap=None,
                type='img'):
    ''' Rotate only images according to its direction 
    
    Args:
        type (str): "img" or "label"
    '''

    # preparation
    os.makedirs(img_dst_dir, exist_ok=True)
    if colormap is None:
        colormap = np.array([[128, 128, 128], [0, 0, 0], [255, 255, 255]])

    # read orientation infos
    orients = fu.read_file_as_list(orient_file)
    orients = {l.split()[0]: int(l.split()[1]) for l in orients}

    # rotate image
    for img_name in os.listdir(img_src_dir):
        if type == 'img':
            img = cv2.imread(osp.join(img_src_dir, img_name), -1)
        else:
            img = read_label_png(osp.join(img_src_dir, img_name), check_path= False)

        timestamp = re.findall(r'2019\d{10}_2019\d{10}', img_name)
        assert len(timestamp) == 1
        timestamp = timestamp[0]

        if orients[timestamp]:
            dst_img = img[::-1, ::-1]
        else:
            dst_img = img
        # save images
        img_dst_path = osp.join(img_dst_dir, img_name)
        if type == 'img':
            cv2.imwrite(img_dst_path, dst_img)
        else:
            lu.lblsave(img_dst_path, dst_img, colormap)
        
        print(f'src img: {osp.join(img_src_dir, img_name)}\ndst img:{img_dst_path}\norient: {orients[timestamp]}')
        print()

        # save images
        if tmp_dir is not None:
            img_ori_path = osp.join(tmp_dir, 'ori.jpg')
            img_dst_path = osp.join(tmp_dir, 'dst.jpg')
            img = Image.fromarray(img)
            dst_img = Image.fromarray(dst_img)
            img.save(img_ori_path)
            dst_img.save(img_dst_path)
            print(f'ori img: {img_ori_path}\ndst img: {img_dst_path}')
            print()


if __name__ == '__main__':
    ''' rotate_img_labels of SN6 full'''
    # img_src_dir = r'data/SN6_full/SAR-PRO'
    # img_dst_dir = r'data/SN6_sup/SAR-PRO_rotated'
    # mask_src_dir = r'data/SN6_sup/label_mask'
    # mask_dst_dir = r'data/SN6_sup/label_mask_rotated'
    # orient_file = r'data/SN6_full/SummaryData/SAR_orientations.txt'
    # colormap = None
    # # tmp_dir = 'tmp2'
    # tmp_dir = None
    # rotate_img_labels(img_src_dir, img_dst_dir, mask_src_dir, mask_dst_dir, orient_file, colormap, tmp_dir)
    
    ''' rotate_imgs of SN6 full'''
    # # img_src_dir = r'data/SN6_full/SAR-ul'
    # # img_dst_dir = r'data/SN6_sup/SAR-ul_rotated'
    # img_src_dir = r'data/SN6_sup/fh_mask'
    # img_dst_dir = r'data/SN6_sup/fh_mask_rotated'
    # orient_file = r'data/SN6_full/SummaryData/SAR_orientations.txt'
    # colormap = None
    # # tmp_dir = 'tmp2'
    # tmp_dir = None
    # rotate_imgs(img_src_dir, img_dst_dir, orient_file, colormap=colormap, tmp_dir=tmp_dir, type='label')
    
    ''' rotate_img_labels of SN6 extend '''
    img_src_dir = r'data/SN6_extend/tile_pauli/900'
    img_dst_dir = r'data/SN6_sup/SAR-extend_rotated'
    mask_src_dir = r'data/SN6_sup/label_mask'
    mask_dst_dir = r'data/SN6_sup/label_mask_rotated'
    orient_file = r'data/SN6_full/SummaryData/SAR_orientations.txt'
    colormap = None
    # tmp_dir = 'tmp2'
    tmp_dir = None
    rotate_img_labels(img_src_dir, img_dst_dir, mask_src_dir, mask_dst_dir, orient_file, colormap, tmp_dir)
    
