'''
Author: Shuailin Chen
Created Date: 2021-10-21
Last Modified: 2021-10-21
	content: calculate the valid ratio of a image of SpaceNet 6 extended dataset
'''

import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import mylib.image_utils as iu
import mylib.file_utils as fu
import matplotlib.pyplot as plt
from copy import deepcopy

from unsup_seg import get_valid_mask


def get_valid_ratio(img_dir, tmp_dir=None):

    valid_ratios = []
    for ii, img_name in enumerate(os.listdir(img_dir)):
        img = cv2.imread(osp.join(img_dir, img_name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bi = gray > 30
        valid_mask =  get_valid_mask(bi, kernel_size=5)
        ratio = valid_mask.sum()/np.prod(valid_mask.shape)
        valid_ratios.append(ratio)

        if tmp_dir is not None:
            iu.save_as_gif([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), valid_mask*255], osp.join(tmp_dir, 'ratio.gif'))
            print(f'{ii}\nreading {osp.join(img_dir, img_name)}, ratio: {ratio}')
            print()
    
    valid_ratios =  np.array(valid_ratios)
    plt.hist(valid_ratios, density=False, bins=100)
    if tmp_dir is not None:
        plt.savefig(osp.join(tmp_dir, 'hist.png'))

    print(f'\n average: {valid_ratios.mean()}\nmax: {valid_ratios.max()}\nmin: {valid_ratios.min()}')


def thresholding_valid_ratio(img_dir: str, thres: float, split_files: list, tmp_dir=None):
    ''' Filtering files whose valid region ratio is greater than defined thredhold 

    Args:
        thres (float): ratio threshold, should in [0, 1]

    Returns:
        split files whose valid region ratio is greater than threshold
    '''
    
    new_split_files = deepcopy(split_files)
    for ii, img_name in enumerate(split_files):
        img = cv2.imread(osp.join(img_dir, img_name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bi = gray > 30
        valid_mask =  get_valid_mask(bi, kernel_size=5)
        ratio = valid_mask.sum()/np.prod(valid_mask.shape)
        
        if ratio < thres:
            new_split_files.remove(img_name)

            if tmp_dir is not None:
                iu.save_as_gif([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), valid_mask*255], osp.join(tmp_dir, 'ratio.gif'))

            print(f'{ii}:\nremoving {osp.join(img_dir, img_name)}\nratio: {ratio}')
            print()
    
    print(f'ori len: {len(split_files)}\nnew len: {len(new_split_files)}')
    return new_split_files


if __name__ == '__main__':
    ''' get_valid_ratio '''
    # img_dir = r'data/SN6_extend/tile_sinclairv2_rotated/900'
    # tmp_dir = r'tmp'
    # get_valid_ratio(img_dir, tmp_dir)

    ''' thresholding_valid_ratio'''
    img_dir = r'data/SN6_extend/tile_sinclairv2_rotated/900'
    tmp_dir = r'tmp'
    thres = 0.2
    split_file_path = r'data/SN6_sup/split/extend_train.txt'
    split_files = fu.read_file_as_list(split_file_path)

    new_split_files = thresholding_valid_ratio(img_dir, thres=thres, split_files=split_files, tmp_dir=tmp_dir)
    fu.write_file_from_list(new_split_files, r'data/SN6_sup/split/extend_train_GT02.txt')