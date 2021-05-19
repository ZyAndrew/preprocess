'''
Author: Shuailin Chen
Created Date: 2021-02-01
Last Modified: 2021-05-19
	content: 
'''
import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import cv2

from mylib import polSAR_utils as psr
from mylib import mathlib

save_dir = r'/home/csl/code/preprocess/tmp'

if __name__ == '__main__':
	path = r'/home/csl/code/preprocess/data/MGGF3jihuanBC20210427/GF3_KAS_QPSI_024808_W116.2_N37.3_20210427_L2_AHV_L20005617967'
	s4 = psr.read_c3_GF3_L2(path, is_print=True)

	# for ii in range(4):
	# 	data = s4[ii, ...]

	# 	# plt.hist(data.flatten(), 256)
	# 	# plt.savefig(osp.join(save_dir, f'{ii}_hist.png'))
	# 	# plt.clf()

	# 	cv2.imwrite(osp.join(save_dir, f'{ii}_img.png'), (mathlib.min_max_map(data)*255).astype(np.uint8))

	# s4 = s4[:, ::4, ::4]
	rgb = psr.rgb_by_s2(s4, type='sinclair', if_log=False, if_mask=True)
	# plt.hist()
	cv2.imwrite(osp.join(path, f'SinclairRGB.png'), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))



