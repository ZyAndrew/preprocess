'''
Author: Shuailin Chen
Created Date: 2021-09-16
Last Modified: 2021-10-19
	content: unsupervised segmentation for spacenet6 extend sinclair v2 images
'''

import os
import os.path as osp
import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
import numpy as np
from numpy import ndarray
import mylib.image_utils as iu
import mylib.labelme_utils as lu
from PIL import Image
from time import time

from merge_duplicate_label import read_label_png


def get_valid_mask(img: ndarray, kernel_size=5):
    ''' Get valid mask of a binary image 
    
    Args:
        kernel_size (int): kernel size of closing operation

    Returns:
        closed (ndarray): valid mask
    '''

    assert img.ndim==2, f'expect #dim of img to be 2, got {img.ndim}'
    assert img.max()<=1, f'expect max value of img to 1, got {img.max()}'
    
    h, w = img.shape
    # 扩展这个mask，因为处于图像边缘的0像素，经过闭运算后会变成1，因此将其边界扩展以避免这样的情况
    tmp = np.zeros((h + 2*kernel_size, w + 2*kernel_size))
    tmp[kernel_size: h+kernel_size, kernel_size: w+kernel_size] = img
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)
    closed = tmp[kernel_size: h+kernel_size, kernel_size:w+kernel_size]
    
    return closed  


def get_FH_mask(img_dir, mask_dir, dst_mask_dir, tmp_dir=None, 
                sigma = 5,
                k = 500,
                min_size = 100):
    ''' Felzenszwalb & Huttenlocher's segmentation method based on graph, 
    http://cs.brown.edu/people/pfelzens/segment/
    '''
    
    img_paths = os.listdir(img_dir)
    print(f'totally {len(img_paths)} images')
    real_segments = []
    for ii, img_path in enumerate(img_paths):
        if osp.splitext(img_path)[1] != '.jpg':
            continue
        img_path = osp.join(img_dir, img_path)
        img = cv2.cvtColor(cv2.imread(img_path, -1), 
                            cv2.COLOR_BGR2RGB)

        # read mask
        if mask_dir is not None:
            ''' Read from label mask '''
            raise NotImplementedError
            # mask_path = img_path.replace(img_dir, mask_dir).replace('tif', 'png') \
            #                     .replace('SAR-Intensity', 'PS-RGB')
            # valid_mask = read_label_png(mask_path, check_path=False)
            # valid_mask = valid_mask > 0
        else:
            ''' Generate from raw image '''
            # due to the blur operation, the threholding value here should set to 30, not 0 in the original SN6
            bi = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 30
            valid_mask = get_valid_mask(bi, kernel_size=5)

        fh_seg = segmentation.felzenszwalb(img, scale=100, sigma=5, min_size=1000)
        fh_seg += 1 #start from 1, 0 for invalid segment

        # mask the invalid regions
        if np.any(valid_mask==0):
            fh_invalid_idx = fh_seg[valid_mask==0]
            val, cnt = np.unique(fh_invalid_idx, return_counts=True)
            invalid_idx = val[np.argmax(cnt)]
            fh_seg[fh_seg==invalid_idx] = 0
            fh_seg[valid_mask==0] = 0
        
        real_segments.append(len(np.unique(fh_seg)))
        print(f'FH actually #segments: {len(np.unique(fh_seg))}')
        dst_mask_path = osp.join(dst_mask_dir, img_path.replace('jpg', 'png')\
                                            .replace(img_dir, dst_mask_dir))
        print(f'{ii}: saving {dst_mask_path}')
        lu.lblsave(dst_mask_path, fh_seg)

        if tmp_dir is not None:
            fh_bound = segmentation.mark_boundaries(img, fh_seg)
            # iu.save_image_by_cv2(img, osp.join(tmp_dir, 'img.jpg'))
            # iu.save_image_by_cv2(valid_mask, osp.join(tmp_dir,
            #                                         'valid_mask.jpg'))
            # iu.save_image_by_cv2(fh_bound, osp.join(tmp_dir, 'FH.jpg'))
            fh_bound = (fh_bound*255).astype(np.uint8)
            valid_mask = (valid_mask*255).astype(np.uint8)
            Image.fromarray(fh_bound).save(osp.join(tmp_dir, 'tmp.gif'), format='GIF', append_images=[Image.fromarray(valid_mask), Image.fromarray(valid_mask)], loop=0, save_all=True, duration=700)
            iu.save_as_gif([img, (valid_mask*255).astype(np.uint8)], osp.join(tmp_dir, 'valid.gif'))
        
        print()


if __name__ == '__main__':
    ''' SN6_full '''
    tmp_dir = r'/home/csl/code/preprocess/tmp2'
    img_dir = r'data/SN6_extend/tile_sinclairv2_rotated/900'
    mask_dir = None
    dst_mask_dir = r'/home/csl/code/preprocess/data/SN6_sup/extend_fh_mask'
    os.makedirs(dst_mask_dir, exist_ok=True)
    tmp_dir = None
    get_FH_mask(img_dir, mask_dir, dst_mask_dir, tmp_dir)