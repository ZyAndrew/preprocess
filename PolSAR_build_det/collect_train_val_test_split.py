'''
Author: Shuailin Chen
Created Date: 2021-04-22
Last Modified: 2021-04-22
	content: 
'''
import os
import os.path as osp

work_dir = r'./PolSAR_build_det'
path = r'/home/ghw/mmsegmentation/data/ade20k/sar_building_gf3/images'
for dir in os.listdir(path):
	with open(osp.join(work_dir, dir+'.txt'), 'w') as f:
		for file in os.listdir(osp.join(path, dir)):
			f.write(file+'\n')

print('done')