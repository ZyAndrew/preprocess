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



if __name__ == '__main__':
    tmp_dir = r'/home/csl/code/preprocess/tmp2'
    img_dir = r'data/SN6_full/SAR-PRO'
    img_dir = r'data/SN6_full_sinclair'
    mask_dir = r'/home/csl/code/preprocess/data/SN6_full_mask'
    n_segments = 100

    # Input data
    # img = data.immunohistochemistry()
    img_path = r'data/SN6_full/SAR-PRO/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804111224_20190804111453_tile_8683.tif'
    img_path = r'data/SN6_full_sinclair/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804111224_20190804111453_tile_8683.tif'
    print(f'reading {img_path}')
    img = cv2.imread(img_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    iu.save_image_by_cv2(img, osp.join(tmp_dir, 'img.jpg'))

    # read mask
    mask_path = img_path.replace(img_dir, mask_dir).replace('tif', 'png') \
                        .replace('SAR-Intensity', 'PS-RGB')
    valid_mask = read_label_png(mask_path, check_path=False)
    valid_mask = valid_mask > 0
    iu.save_image_by_cv2(valid_mask, osp.join(tmp_dir, 'valid_mask.jpg'))


    # Compute a mask
    lum = color.rgb2gray(img)
    # iu.save_image_by_cv2(lum, osp.join(tmp_dir, 'gray.jpg'))
    # mask = morphology.remove_small_holes(
    #         morphology.remove_small_objects(lum < 0.7, 500), 500)
    # iu.save_image_by_cv2(mask, osp.join(tmp_dir, 'rm_small_hole.jpg'))

    # mask = morphology.opening(mask, morphology.disk(3))
    # iu.save_image_by_cv2(mask, osp.join(tmp_dir, 'opening.jpg'))

    # SLIC result
    # slic = segmentation.slic(img, n_segments=n_segments, start_label=1)
    # slic_bound = segmentation.mark_boundaries(img, slic)
    # slic_bound_mask = segmentation.mark_boundaries(slic_bound, mask, color=(1, 0, 0))
    # iu.save_image_by_cv2(slic_bound, osp.join(tmp_dir, 'slic_mask.jpg'))

    # maskSLIC result
    m_slic = segmentation.slic(img, n_segments=n_segments, mask=valid_mask, start_label=1, convert2lab=True, sigma=5)
    m_slic_bound = segmentation.mark_boundaries(img, m_slic)
    # m_slic_bound_mask = segmentation.mark_boundaries(m_slic_bound, mask, color=(1, 0, 0))
    print(f'actually #segments: {len(np.unique(m_slic))}')
    iu.save_image_by_cv2(m_slic_bound, osp.join(tmp_dir, 'mslic_mask.jpg'))
