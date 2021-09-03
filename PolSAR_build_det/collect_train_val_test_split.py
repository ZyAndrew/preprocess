'''
Author: Shuailin Chen
Created Date: 2021-04-22
Last Modified: 2021-09-03
	content: 将郭浩文分类号的训练、验证、测试集总结成 .txt 文件的格式
'''
import os
import os.path as osp

work_dir = r'./PolSAR_build_det'
path = r'data/ade20k/sar_building_rs2/images'
for dir in os.listdir(path):
	with open(osp.join(work_dir, dir+'.txt'), 'w') as f:
		for file in os.listdir(osp.join(path, dir)):
			f.write(file+'\n')

print('done')