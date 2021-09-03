'''
Author: Shuailin Chen
Created Date: 2021-08-20
Last Modified: 2021-08-20
	content: 
'''

import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt

import mylib.labelme_utils as lu

TMP_FOLDER = 'tmp'


def convert_gt_idx(src_path, dst_path):
    ''' Convert GT image file into index file for ease of mmsegmentation
    '''
    
    os.makedirs(dst_path, exist_ok=True)
    num_samples = len(os.listdir(src_path))//3

    for ii in range(1, num_samples+1):
        meta_path = osp.join(src_path, f'{ii}_')
        label_1 = cv2.imread(meta_path + '1_label.png', -1)
        label_2 = cv2.imread(meta_path + '2_label.png', -1)
        cd_label = cv2.imread(meta_path + 'change.png', -1)

        assert label_1.shape == label_2.shape == cd_label.shape == (512, 512)

        label_1[label_1>0] = 1
        label_2[label_2>0] = 1
        cd_label[cd_label>0] = 1

        # show lables 
        # plt.matshow(label_1)
        # plt.savefig(osp.join(TMP_FOLDER, 'label_1.png'))
        # plt.clf()
        # plt.matshow(label_2)
        # plt.savefig(osp.join(TMP_FOLDER, 'label_2.png'))
        # plt.clf()
        # plt.matshow(cd_label)
        # plt.savefig(osp.join(TMP_FOLDER, 'chagne.png'))
        # plt.clf()

        meta_path = osp.join(dst_path, f'{ii}_')
        lu.lblsave(meta_path + '1_label.png', label_1, np.array([[0, 0, 0], [255, 255, 255]]))
        lu.lblsave(meta_path + '2_label.png', label_2, np.array([[0, 0, 0], [255, 255, 255]]))
        lu.lblsave(meta_path + 'change.png', cd_label, np.array([[0, 0, 0], [255, 255, 255]]))


if __name__ == '__main__':
    src_path = r'data/gaofen/trainData/gt'
    dst_path = r'data/gaofen/trainData/gt_idx'
    convert_gt_idx(src_path, dst_path)