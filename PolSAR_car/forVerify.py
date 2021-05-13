'''
Author: Shuailin Chen
Created Date: 2021-05-12
Last Modified: 2021-05-13
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
    path = r'/home/csl/code/preprocess/data/SN6_extend/newSLC/sar_mag_pol_20190822072404_20190822072642.tif'
    t = tifffile.imread(path)
    print(f'shape: {t.shape}\n dtype: {t.dtype}\n type: {type(t)}')
    im = psr.rgb_by_s2(t.transpose(2, 0, 1))
    im = (im*255).astype(np.uint8)
    cv2.imwrite(osp.join(TMP_DIR, 'pauli.png'), im)