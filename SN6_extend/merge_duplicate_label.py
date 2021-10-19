'''
Author: Shuailin Chen
Created Date: 2021-10-18
Last Modified: 2021-10-19
	content: merge duplicated labels that originally appear both in train and val split, and assign it to train split
'''

import os
import os.path as osp
import re
from PIL import Image
import mylib.labelme_utils as lu
import mylib.file_utils  as fu
import mylib.image_utils as iu
import numpy as np
from copy import deepcopy
import json


def read_label_png(src_path:str, check_path=False)->np.ndarray:
    '''读取 label.png 包含的label信息，这个文件的格式比较复杂，直接读取会有问题，需要特殊处理

    Args:
    src_path (str): label文件路径或者其文件夹
    check_path (bool): 是否检查路径. Default: True

    Returns
    label_idx (ndarray): np.ndarray格式的label信息
    label_name (tuple): tuple 格式的label名字，对应 label_idx里面的索引
    '''
    
    # read label.png, get the label index
    if check_path and src_path[-10:] != r'/label.png' \
                  and src_path[-10:] != r'\label.png':
        src_path = os.path.join(src_path, 'label.png')
    tmp = Image.open(src_path)
    label_idx = np.asarray(tmp)
    return label_idx


def merge_duplicate_labels(label_dir, mask_dir, rgb_dir, slc_dir, tmp_dir=None, opacity=0.5):

	# read label file names
	split_files = {}
	for split in os.listdir(mask_dir):
		if not osp.isdir(osp.join(mask_dir, split)):
			continue
		split_files[split] = []
		for geojson in os.listdir(osp.join(mask_dir, split)):
			crs = re.findall(r'\d{6}_\d{7}', geojson)
			assert len(crs) == 1
			split_files[split].append(crs[0])

	# check duplicate, and assign it to train split
	mask_train_split = fu.read_file_as_list(osp.join(mask_dir,
													'valid_train.txt'))
	mask_val_split = fu.read_file_as_list(osp.join(mask_dir, 'valid_val.txt'))
	slc_train_split = fu.read_file_as_list(osp.join(slc_dir,
													'valid_train.txt'))
	slc_val_split = fu.read_file_as_list(osp.join(slc_dir, 'valid_val.txt'))
	ii = 0
	for crs in split_files['val']:
		if crs in split_files['train']:
			label_train_path = osp.join(label_dir, 'train', f'geoms_{crs}.geojson')
			label_val_path = label_train_path.replace('train', 'val')
			mask_train_path = osp.join(mask_dir, 'train', f'geoms_{crs}.png')
			mask_val_path = mask_train_path.replace('train', 'val')
			rgb_filepath = osp.join(rgb_dir, f'SN6_AOI_11_Rotterdam_PS-RGB_TrainVal_{crs}.tif')

			# merge mask
			mask_train = read_label_png(mask_train_path, check_path=False)
			mask_val = read_label_png(mask_val_path, check_path=False)
			new_mask = mask_train + mask_val
			colormap = np.array([[0, 0, 0], [255, 255, 255]])
			lu.lblsave(mask_train_path, new_mask, colormap=colormap)
			# os.remove(label_val_path)	# not remove

			# merge geojson label
			with open(label_train_path, 'r') as f:
				label_train = json.load(f)
			with open(label_val_path, 'r') as f:
				label_val = json.load(f)
			new_label = deepcopy(label_train)
			new_label['features'].extend(label_val['features'])
			assert len(new_label['features']) == len(label_val['features']) + len(label_train['features'])
			with open(label_train_path, 'w+') as f:
				json.dump(new_label, f)

			print(f'{ii}: \nrgb: {rgb_filepath}\nmask train: {mask_train_path}\nmask val: {mask_val_path}\nlabel train: {label_train_path}\nlabel val: {label_val_path}')
			ii += 1

			# modify train and val split files of mask
			mask_val_split.remove(f'geoms_{crs}.png')
			print(f'rm geoms_{crs}.png in mask val split')
			if f'geoms_{crs}.png' not in mask_train_split:
				mask_train_split.append(f'geoms_{crs}.png')
				print(f'add geoms_{crs}.png in mask train split')
			else:
				print(f'geoms_{crs}.png already exist in mask train split')

			# modify train and val split files of SLC
			rm_slc = []
			for slc in slc_val_split:
				if crs in slc:
					slc_val_split.remove(slc)
					rm_slc.append(slc)
					print(f'rm {slc} in SLC val split')
			for slc in rm_slc:
				if slc not in slc_train_split:
					slc_train_split.append(slc)
					print(f'add {slc} in SLC train split')
				else:
					print(f'{slc} alread exist in SLC train split')
			
			if tmp_dir is not None:
				lu.lblsave(osp.join(tmp_dir, 'new_mask.png'), new_mask, colormap)
				with open(osp.join(tmp_dir, 'label.geojson'), 'w+') as f:
					json.dump(new_label, f)
				mask_gif_path = osp.join(tmp_dir, 'mask.gif')
				iu.save_as_gif([mask_train*255, mask_val*255, new_mask*255], mask_gif_path)
				rgb_gif_path = osp.join(tmp_dir, 'rgb.gif')
				rgb = Image.open(rgb_filepath)
				rgb = np.asarray(rgb).astype(np.uint8)
				new_mask = np.tile(new_mask[..., None], (1, 1, 3)).astype(np.uint8)
				mixed = opacity * rgb + (1-opacity) * new_mask*255
				iu.save_as_gif([mixed.astype(np.uint8), rgb], rgb_gif_path)
				print(f'mask gif: {mask_gif_path}\nRGB gif: {rgb_gif_path}')

	fu.write_file_from_list(mask_train_split, osp.join(mask_dir, 'valid_train.txt'))
	fu.write_file_from_list(mask_val_split, osp.join(mask_dir, 'valid_val.txt'))
	fu.write_file_from_list(slc_train_split, osp.join(slc_dir, 'valid_train.txt'))
	fu.write_file_from_list(slc_val_split, osp.join(slc_dir, 'valid_val.txt'))


if __name__ == '__main__':
	label_dir = r'data/SN6_extend/tile_label/900'
	mask_dir = r'data/SN6_extend/tile_mask/900'
	rgb_dir = r'data/SN6_extend/tile_rgb/900'
	slc_dir = r'data/SN6_extend/tile_slc/900'
	tmp_dir = r'tmp'
	merge_duplicate_labels(label_dir, mask_dir, rgb_dir, slc_dir, tmp_dir)
