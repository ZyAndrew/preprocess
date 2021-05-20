'''
Author: Shuailin Chen
Created Date: 2021-05-10
Last Modified: 2021-05-20
  content: data post-processing
'''

import os
import os.path as osp
import glob
import numpy as np
import tifffile
import cv2

from mylib import polSAR_utils as psr
from mylib import file_utils as fu



''' s2 to c3 of SN6 data '''
path = r'/home/csl/code/preprocess/PolSAR_car/data/car/SN6_extend'
for tm in os.listdir(path):
    for idx in os.listdir(osp.join(path, tm)):
        current_path = osp.join(path, tm, idx)
        s2 = psr.read_s2(current_path)
        C3 = psr.s22c3(s2=s2)
        dst_path = osp.join(current_path, 'C3')
        psr.write_c3(dst_path, C3, is_print=True)
        pauli = psr.rgb_by_c3(C3)
        cv2.imwrite(osp.join(dst_path, 'PauliRGB.png'), cv2.cvtColor((255*pauli).astype(np.uint8), cv2.COLOR_BGR2RGB))


''' extract ROI area from GF3 data'''
# dst_folder = r'/home/csl/code/preprocess/PolSAR_car/data/ship'
# paren_folder = r'/home/csl/code/preprocess/data/SAR_CD/GF3/data'
# locations = [r'E130_N34_日本鞍手/降轨/1', r'E139_N35_日本横滨/降轨/1']
# times = [[20190602, 20170607, 20190602], [20190615]]
# rois = [[[3660, 2491, 49, 74], [3743, 2426, 69, 66], [3812, 2449, 62, 58]], [[1436, 807, 185, 137]]]
# for loc, times_, rois_ in zip(locations, times, rois):
#     for idx, (tm, roi) in enumerate(zip(times_, rois_)):
#         src_path = osp.join(paren_folder, loc, str(tm))
#         idx = idx // 2
#         dst_path = osp.join(dst_folder, loc, str(tm), str(idx+1))
#         psr.exact_patch_s2(src_path, roi, dst_path)


print('done')
