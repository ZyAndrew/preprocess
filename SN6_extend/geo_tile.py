'''
Author: Shuailin Chen
Created Date: 2021-10-18
Last Modified: 2021-10-19
	content: tile the train-test split according to geo-coordinates specified by 王宇轩, not by the official intructions
'''

from osgeo import gdal
import os
import os.path as osp
import mylib.file_utils as fu

from merge_duplicate_label import read_label_png


def geo_tile_SN6_extend(ori_dir, new_dir, geo_min_thres):
    
    proj_list = []
    for file in os.listdir(ori_dir):
        data = gdal.Open(osp.join(ori_dir, file))
        proj = data.GetGeoTransform()
        proj_list.append(proj[0])

    proj_list = sorted(proj_list)
    geo_max_thres = proj_list[-1]

    train_list, test_list = [], []
    for file in os.listdir(new_dir):
        if not file.endswith('.tif'):
            continue

        data = gdal.Open(osp.join(new_dir, file))
        proj = data.GetGeoTransform()
        if proj[0] <= geo_min_thres and proj[0] >= geo_max_thres:
            train_list.append(file)
        else:
            test_list.append(file)
    print(f'max geo thres: {geo_max_thres}\nmin geo thres: {geo_min_thres}\n#train: {len(train_list)}\n#test: {len(test_list)}\nall: {len(train_list)+len(test_list)}')

    return train_list, test_list


if __name__ == '__main__':
    ori_dir = r'data/SN6_full/SAR-Intensity'
    new_dir = r'data/SN6_extend/tile_slc/900'
    # SAR_dir = r'data/SN6_full/SAR-Intensity'
    split_dir = r'data/SN6_sup/split'
    geo_min_thres = 594906.9818506762
    train_list, test_list = geo_tile_SN6_extend(ori_dir, new_dir,
                                                geo_min_thres=geo_min_thres)

    fu.write_file_from_list(train_list, osp.join(split_dir, 
                                                'extend_train.txt'))
    fu.write_file_from_list(test_list, osp.join(split_dir, 
                                                'extend_test.txt'))