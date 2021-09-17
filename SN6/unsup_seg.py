'''
Author: Shuailin Chen
Created Date: 2021-09-16
Last Modified: 2021-09-17
	content: unsupervised segmentation for spacenet6 intensity images
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
import mylib.image_utils as iu
import mylib.labelme_utils as lu
from PIL import Image
from time import time

from ImageSegmentation import seg


def read_label_png(src_path:str, check_path=True)->np.ndarray:
    '''读取 label.png 包含的label信息，这个文件的格式比较复杂，直接读取会有问题，需要特殊处理

    Args:
    src_path (str): label文件路径或者其文件夹
    check_path (bool): 是否检查路径. Default: True

    Returns
    label_idx (ndarray): np.ndarray格式的label信息
    label_name (tuple): tuple 格式的label名字，对应 label_idx里面的索引
    '''
    
    # read label.png, get the label index
    if check_path and src_path[-10:] != r'/label.png' \
                  and src_path[-10:] != r'\label.png':
        src_path = os.path.join(src_path, 'label.png')
    tmp = Image.open(src_path)
    label_idx = np.asarray(tmp)
    return label_idx


def get_slic_mask(img_dir, dst_mask_dir, tmp_dir=None, n_segments=100):

    img_paths = os.listdir(img_dir)
    print(f'totally {len(img_paths)} images')
    real_segments = []
    for ii, img_path in enumerate(img_paths):
        if osp.splitext(img_path)[1] != '.tif':
            continue
        img_path = osp.join(img_dir, img_path)
        img = cv2.cvtColor(cv2.imread(img_path, -1), 
                            cv2.COLOR_BGR2RGB)

        # read mask
        mask_path = img_path.replace(img_dir, mask_dir).replace('tif', 'png') \
                            .replace('SAR-Intensity', 'PS-RGB')
        valid_mask = read_label_png(mask_path, check_path=False)
        valid_mask = valid_mask > 0

        # maskSLIC
        m_slic = segmentation.slic(img, n_segments=n_segments,
                                    mask=valid_mask, start_label=1, convert2lab=True, sigma=5)
        m_slic_bound = segmentation.mark_boundaries(img, m_slic)

        real_segments.append(len(np.unique(m_slic)))
        print(f'{ii}: SLIC actually #segments: {len(np.unique(m_slic))}')
        
        dst_mask_path = osp.join(dst_mask_dir, img_path.replace('tif', 'png')\
                                            .replace(img_dir, dst_mask_dir))
        print(f'{ii}: saving {dst_mask_path}')
        iu.save_image_by_cv2(m_slic, dst_mask_path)

        if tmp_dir is not None:
            iu.save_image_by_cv2(img, osp.join(tmp_dir, 'img.jpg'))
            iu.save_image_by_cv2(valid_mask, osp.join(tmp_dir,
                                                    'valid_mask.jpg'))
            iu.save_image_by_cv2(m_slic_bound, osp.join(tmp_dir,
                                                    'mslic_mask.jpg'))

    assert len(real_segments) == len(img_paths)
    real_segments = np.array(real_segments)
    print(f'#segments: mean: {real_segments.mean()}, std: {real_segments.std()}')
        


if __name__ == '__main__':
    tmp_dir = r'/home/csl/code/preprocess/tmp2'
    img_dir = r'data/SN6_full/SAR-PRO'
    mask_dir = r'/home/csl/code/preprocess/data/SN6_sup/label_mask'
    dst_mask_dir = r'/home/csl/code/preprocess/data/SN6_sup/slic_mask'
    n_segments = 100

    get_slic_mask(img_dir, dst_mask_dir, tmp_dir=tmp_dir,
                n_segments=n_segments)

    # FH
    # sigma = 5
    # k = 500
    # min_size = 100
    # start_time = time()
    # fh_seg = seg(img, sigma, k, min_size, osp.join(tmp_dir, 'FH_mask.jpg')) + 1   # start from 1
    # end_time = time()
    # fh_bound = segmentation.mark_boundaries(img, fh_seg)
    # print(f'FH actually #segments: {len(np.unique(fh_seg))}, elapsed time: {end_time-start_time}')
    # iu.save_image_by_cv2(fh_bound, osp.join(tmp_dir, 'FH.jpg'))