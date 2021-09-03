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

TMP_FOLDER = 'tmp'

def verify_build_and_change_label(path: str):
    ''' Verify whether the change detection label is the result of xor of two building labels
    '''

    num_samples = len(os.listdir(path))//3
    for ii in range(1, num_samples+1):
        meta_path = osp.join(path, f'{ii}_')
        label_1 = cv2.imread(meta_path + '1_label.png', -1)
        label_2 = cv2.imread(meta_path + '2_label.png', -1)
        cd_label = cv2.imread(meta_path + 'change.png', -1)

        assert label_1.shape == label_2.shape == cd_label.shape == (512, 512)

        # show lables 
        plt.matshow(label_1)
        plt.savefig(osp.join(TMP_FOLDER, 'label_1.png'))
        plt.clf()
        plt.matshow(label_2)
        plt.savefig(osp.join(TMP_FOLDER, 'label_2.png'))
        plt.clf()
        plt.matshow(cd_label)
        plt.savefig(osp.join(TMP_FOLDER, 'chagne.png'))
        plt.clf()

        label_1 = label_1.astype(np.int)
        label_2 = label_2.astype(np.int)
        cd_label = cd_label.astype(np.int)
        direct_cd_label = np.abs(label_1-label_2)

        plt.matshow(direct_cd_label)
        plt.savefig(osp.join(TMP_FOLDER, 'd_chagne.png'))
        plt.clf()
        # plt.matshow(cd_label)
        # plt.savefig(osp.join(TMP_FOLDER, 'chagne.png'))
        # plt.clf()

        assert np.all(direct_cd_label == cd_label)


if __name__ == '__main__':
    path = r'data/gaofen/trainData/gt'
    verify_build_and_change_label(path)