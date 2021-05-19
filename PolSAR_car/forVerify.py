'''
Author: Shuailin Chen
Created Date: 2021-05-12
Last Modified: 2021-05-14
	content: 
'''
import os
import os.path as osp

import numpy as np
import tifffile
import cv2

import solaris.preproc.image as image
from mylib import polSAR_utils as psr

TMP_DIR = r'/home/csl/code/preprocess/tmp'



if __name__ == '__main__':
    # path = r'/home/csl/code/preprocess/data/SN6_extend/newSLC/sar_mag_pol_20190822072404_20190822072642.tif'
    # t = tifffile.imread(path)
    # print(f'shape: {t.shape}\n dtype: {t.dtype}\n type: {type(t)}')
    # im = psr.rgb_by_s2(t.transpose(2, 0, 1))
    # im = (im*255).astype(np.uint8)
    # cv2.imwrite(osp.join(TMP_DIR, 'pauli.png'), im)


    path = r'/home/csl/code/preprocess/PolSAR_car/data/car/20190822093113_20190822093410/1'
    save_dir = r'/home/csl/code/preprocess/tmp'
    s2 = psr.read_s2(path)
    pauli = psr.rgb_by_s2(s2)
    cv2.imwrite(osp.join(save_dir, 'pauli.png'), (pauli*255).astype(np.uint8))

    
    # path = r'/home/csl/code/preprocess/PolSAR_car/ship/E139_N35_日本横滨/降轨/1/20190615/C3'
    # save_dir = r'/home/csl/code/preprocess/tmp'
    # c3 = psr.read_c3(path)
    # pauli = psr.rgb_by_c3(c3)
    # cv2.imwrite(osp.join(save_dir, 'pauli.png'), (pauli*255).astype(np.uint8))