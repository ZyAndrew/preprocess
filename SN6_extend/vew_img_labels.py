'''
Author: Shuailin Chen
Created Date: 2021-10-18
Last Modified: 2021-10-18
	content: view the image and label (mask), through observation, can conclude that the relative position of label and image is differet in case of different directions
'''

import os
import os.path as osp
import cv2
import tifffile
from PIL import Image
import mylib.file_utils as fu
import mylib.labelme_utils as lu
import mylib.image_utils as iu
import numpy as np
import re
from glob import glob


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
    mask_dir = r'data/SN6_extend/tile_mask/900'
    pauli_dir = r'data/SN6_extend/tile_pauli/900'
    orient_file = r'data/SN6_extend/SummaryData/SAR_orientations.txt'
    tmp_dir = r'tmp'
    opacity = 0.5

    orients = fu.read_file_as_list(orient_file)
    orients = {l.split()[0]: int(l.split()[1]) for l in orients}

    splits_name = ['train', 'val']
    split = dict()
    for sp in splits_name:
        split[sp] = fu.read_file_as_list(osp.join(mask_dir,
                                                f'valid_{sp}.txt'))
    # rotate image
    for pauli_name in os.listdir(pauli_dir):

        # read PauliRGB
        pauli = cv2.imread(osp.join(pauli_dir, pauli_name))
        pauli = cv2.cvtColor(pauli, cv2.COLOR_RGB2BGR)

        # read mask
        geo_coord = re.findall(r'_(\d{6}_\d{7})\.jpg', pauli_name)
        assert len(geo_coord) == 1
        geo_coord = geo_coord[0]
        mask_name = f'geoms_{geo_coord}.png'
        if mask_name in split['train']:
            mask_path = osp.join(mask_dir, 'train', mask_name)
        else:
            mask_path = osp.join(mask_dir, 'val', mask_name)
        mask = read_label_png(mask_path, check_path=False)
        mask = (mask > 0)*255
        mask = np.tile(mask[..., None], (1, 1, 3)).astype(np.uint8)
        mixed = (opacity * pauli + (1-opacity) * mask).astype(np.uint8)

        # save images
        mixed_path = osp.join(tmp_dir, 'mixed.png')
        gif_path = osp.join(tmp_dir, 'tt.gif')
        iu.save_image_by_cv2(mixed, mixed_path)
        
        pauli = Image.fromarray(pauli)
        mask = Image.fromarray(mask)
        mixed = Image.fromarray(mixed)
        pauli.save(gif_path, format='GIF', append_images=[mixed], loop=0, save_all=True, duration=700)

        # print infos
        timestamp = re.findall(r'2019\d{10}_2019\d{10}', pauli_name)
        assert len(timestamp) == 1
        timestamp = timestamp[0]
        print(f'pauli: {osp.join(pauli_dir, pauli_name)}\nmask: {osp.join(mask_dir, mask_name)}\nmixed: {mixed_path}\ngif: {gif_path}\norient: {orients[timestamp]}')
        print()
