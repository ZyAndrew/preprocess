'''
Author: Shuailin Chen
Created Date: 2021-09-17
Last Modified: 2021-09-17
	content: 
'''
import os
import os.path as osp

import torch
import numpy as np 
import tifffile
from glob import glob
import matplotlib.pyplot as plt
from mylib import labelme_utils as lu 
import mylib.image_utils as iu
from mylib import polSAR_utils as psr
import re
import cv2
import xml.etree.ElementTree as et
import h5py


if __name__ == '__main__':
    path = r'/home/csl/code/preprocess/data/csl_wuhan/1028584-816233/CSKS3_SCS_B_HI_01_HH_RA_SF_20200922220155_20200922220203.h5'

    intensity = psr.read_csk_L1A_as_intensity(path, is_print=True)
    gray = psr.gray_by_intensity(intensity, type='log', if_mask=True, is_print=True)

    # f = h5py.File(path, 'r')
    # print(f)

    # keys = list(f.keys())[0]
    # print(f'keys: {f.keys()}') 

    # data = f[keys]

    # a = data['B001']    # none    
    # b = data['QLK']     #quick look
    # c = data['SBI']
    
    # cd = c[()]
    # cd = cd.astype(np.float32)
    # mcd = cd[..., 0]**2+cd[..., 1]**2
    # print(f'shape: {mcd.shape}, max: {mcd.max()}, min: {mcd.min()}')

    # gray = psr.gray_by_intensity(mcd, type='log', if_mask=True, is_print=True)
    iu.save_image_by_cv2(gray, dst_path=osp.join(r'/home/csl/code/preprocess/tmp2', 'gray2.jpg'))