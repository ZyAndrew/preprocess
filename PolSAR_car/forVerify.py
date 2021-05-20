'''
Author: Shuailin Chen
Created Date: 2021-05-12
Last Modified: 2021-05-20
	content: 
'''
import os
import os.path as osp

import numpy as np
import tifffile
import cv2

# import solaris.preproc.image as image
from mylib import polSAR_utils as psr

TMP_DIR = r'/home/csl/code/preprocess/tmp'



if __name__ == '__main__':
    path = r'/home/csl/code/preprocess/PolSAR_car/data/ship/E139_N35_日本横滨/降轨/1/20190615/1'
    save_path = r'/home/csl/code/preprocess/tmp'
    data = psr.read_s2(path)
    pauli = psr.rgb_by_s2(data)
    cv2.imwrite(osp.join(save_path, 'pauli.jpg'), cv2.cvtColor(pauli, cv2.COLOR_BGR2RGB))

    
    # path = path.replace('s2', 'C3')
    path = r'/home/csl/code/preprocess/PolSAR_car/data/ship/E139_N35_日本横滨/降轨/1/20190615/C3'
    data = psr.read_c3(path)
    pauli = psr.rgb_by_c3(data)
    cv2.imwrite(osp.join(save_path, 'pauli2.jpg'), cv2.cvtColor((255*pauli).astype(np.uint8), cv2.COLOR_BGR2RGB))
    