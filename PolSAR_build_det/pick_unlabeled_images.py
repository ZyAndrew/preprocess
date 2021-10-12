'''
Author: Shuailin Chen
Created Date: 2021-09-03
Last Modified: 2021-09-04
	content: 
    NOTE: undone
'''

import os
import os.path as osp
from glob import glob
import mylib.file_utils as fu

def pick_unlabeled_images(imgs_path, split_path, save_file=None):
    ''' Pick the unlabeled images '''

    if save_file is None:
        save_file = r'unlabeled.txt'

    # read split files
    labeled_list = []
    for split in glob(osp.join(split_path, '*.txt')):
        # with open(osp.join(split_path, split), 'r') as f:
        labeled_list += fu.read_file_as_list(split)

    # check all the images, if the image is not in the split files, add it to the unlabeld_list
    unlabeled_list = []
    for loc in os.listdir(imgs_path):
        for tm in os.listdir(osp.join(imgs_path, loc)):
            for num in os.listdir(osp.join(imgs_path, loc, tm, 'C3')):
                if osp.isfile(osp.join(imgs_path, loc, tm, num)):
                    continue

                patch_name = f'{loc}{tm}_{int(num):03d}'
                unlabeled_list.append(patch_name)
    
    # write unlabeled.txt
    fu.write_file_from_list(unlabeled_list)


if __name__ == '__main__':
    imgs_path = r'data/PolSAR_building_det/data/RS2'
    split_path = r'data/ade20k/split/RS2'
    pick_unlabeled_images(imgs_path, split_path, )