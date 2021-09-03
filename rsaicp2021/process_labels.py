'''
Author: Shuailin Chen
Created Date: 2021-06-16
Last Modified: 2021-06-29
	content: 
'''

import os.path as osp
import os
import cv2
import numpy as np

import mylib.file_utils as fu
import mylib.labelme_utils as lu

def merge_labels(src_path):
    ''' Merge two labels image into a image'''
    for split in os.listdir(src_path):

        if not osp.isdir(osp.join(src_path,split)):
            continue

        dst_path = osp.join(src_path, split, 'label')
        fu.mkdir_if_not_exist(dst_path)
        label1_src_path = osp.join(src_path, split, 'label1')
        label2_src_path = osp.join(src_path, split, 'label2')
        for img in os.listdir(label1_src_path):
            label1_img = cv2.imread(osp.join(label1_src_path, img))
            label2_img = cv2.imread(osp.join(label2_src_path, img))
            label_merged = label1_img + label2_img
            cv2.imwrite(osp.join(dst_path, img), label_merged)


def RGB_annotation_to_index(src_path, palette):
    ''' Map RGB annotation image into index image'''

    for split in os.listdir(src_path):

        if not osp.isdir(osp.join(src_path,split)):
            continue

        dst_path = osp.join(src_path, split, 'label_index')
        fu.mkdir_if_not_exist(dst_path)
        label_src_path = osp.join(src_path, split, 'label')
        for img in os.listdir(label_src_path):
            label = cv2.imread(osp.join(label_src_path, img))
            label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
            label_idx = 255*np.ones(label.shape[:2], dtype=np.uint8)

            for idx, color in enumerate(palette):
                color = np.array(color)[None, None, ...]
                label_idx[(label==color).all(axis=2)] = idx
            
            print(f'processing: {osp.join(label_src_path, img)}')
            lu.lblsave(osp.join(dst_path, img), label_idx, np.array(palette))



if __name__ == '__main__':
    src_path = r'/home/csl/code/preprocess/data/S2Looking'
    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 0, 255]]
    # merge_labels()
    RGB_annotation_to_index(src_path, PALETTE)