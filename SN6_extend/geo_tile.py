'''
Author: Shuailin Chen
Created Date: 2021-10-18
Last Modified: 2021-10-18
	content: tile the train-test split according to geo-coordinates specified by 王宇轩, not by the official intructions
'''

from osgeo import gdal
import os
import os.path as osp
import mylib.file_utils as fu

def geo_tile_SN6_extend(SAR_dir, geo_thres):

    train_list, test_list = [], []
    for file in os.listdir(SAR_dir):
        if not file.endswith('.tif'):
            continue

        data = gdal.Open(osp.join(SAR_dir, file))
        proj = data.GetGeoTransform()
        if proj[0] <= geo_thres:
            train_list.append(file)
        else:
            test_list.append(file)
    print(f'#train: {len(train_list)}\n#test: {len(test_list)}\nall: {len(train_list)+len(test_list)}')

    return train_list, test_list


if __name__ == '__main__':
    SAR_dir =r'data/SN6_extend/tile_slc/900'
    # SAR_dir = r'data/SN6_full/SAR-Intensity'
    split_dir = r'data/SN6_sup/split'
    geo_thres = 594906.9818506762
    train_list, test_list = geo_tile_SN6_extend(SAR_dir,
                                                geo_thres=geo_thres)

    fu.write_file_from_list(train_list, osp.join(split_dir, 
                                                'extend_train.txt'))
    fu.write_file_from_list(test_list, osp.join(split_dir, 
                                                'extend_test.txt'))