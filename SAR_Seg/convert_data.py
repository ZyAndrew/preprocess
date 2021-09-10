'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-10
	content: convert the original one channel per image into 4/3 channel per image
'''

import os
from PIL.Image import merge
import cv2
import numpy as np
from numpy import ndarray
import natsort
import mylib
import os.path as osp


def min_max_map(x):
    ''' Map all the elements of x into [0,1] using min max map
    @in     x   ndarray
    @out        ndarray
    '''
    min = x.reshape(1,-1).min()
    max = x.reshape(1,-1).max()
    return (x-min)/(max-min)


def min_max_contrast_median(data:ndarray):
    ''' Use the iterative method to get special min and max value
        @out    - min and max value in a tuple
    '''
    # remove nan and inf, vectorization
    data = data.reshape(1,-1)
    data = data[~(np.isnan(data) | np.isinf(data))]

    # iterative find the min and max value
    med = np.median(data)
    med1 = med.copy()       # the minimum value
    med2 = med.copy()       # the maximum value
    for ii in range(3):
        part_min = data[data<med1]
        if part_min.size>0:
            med1 = np.median(part_min)
        else:
            break
    for ii in range(3):
        part_max = data[data>med2]
        if part_max.size>0:
            med2 = np.median(part_max)
        else:
            break
    return med1, med2


def min_max_contrast_median_map(data:ndarray)->ndarray:
    '''
    @brief  -map all the elements of x into [0,1] using        
            min_max_contrast_median function
    @out    -the nomalized ndarray
    '''
    min, max = min_max_contrast_median(data[data != 10*np.log10(np.finfo(float).eps)])
    # print('ggg   ', min, max, 'ggg')
    return np.clip((data-min)/(max - min), a_min=0, a_max=1)


def band3_data_pre_psp(src_path:str, out_path:str):
    ''' 仿 PolSAR pro 的 pauliRGB 的预处理方法 '''
    data_dir = src_path
    file_list = os.listdir(data_dir)
    data_list = []
    HH_data_list = []
    HV_data_list = []
    VV_data_list = []
    for f in file_list:
        if f[-1] == "f":
            data_list.append(f)
    for data in data_list:
        if 'HH' in data:
            HH_data_list.append(data)
        elif 'HV' in data:
            HV_data_list.append(data)
        elif 'VV' in data:
            VV_data_list.append(data)
    merge_data_3band_list = []
    file_name_list = []
    HH_data_list = natsort.natsorted(HH_data_list, alg=natsort.PATH)
    HV_data_list = natsort.natsorted(HV_data_list, alg=natsort.PATH)
    VV_data_list = natsort.natsorted(VV_data_list, alg=natsort.PATH)

    for i in range(len(HH_data_list)):
        imgHH = cv2.imread(osp.join(data_dir, HH_data_list[i]), -1)
        imgHV = cv2.imread(osp.join(data_dir, HV_data_list[i]), -1)
        imgVV = cv2.imread(osp.join(data_dir, VV_data_list[i]), -1)

        # incase the datavalue equals to 0
        imgHH[imgHH<1] = 1
        imgHV[imgHV<1] = 1
        imgVV[imgVV<1] = 1

        imgHH = 10*np.log10(imgHH)
        imgHV = 10*np.log10(imgHV)
        imgVV = 10*np.log10(imgVV)

        imgHH = 255*min_max_contrast_median_map(imgHH)
        imgHV = 255*min_max_contrast_median_map(imgHV)
        imgVV = 255*min_max_contrast_median_map(imgVV)

        merge_data_3band = cv2.merge([imgHH, imgHV, imgVV])
        merge_data_3band.astype(np.uint8)
        file_name = HH_data_list[i].replace('_HH', '').replace('.tiff', '.png')

        cv2.imwrite(osp.join(out_path, file_name), merge_data_3band)
        print('write ', file_name, ' succeed')


def band4_data_pre_psp(src_path:str, out_path:str):
    ''' 仿 PolSAR pro 的 pauliRGB 的预处理方法 '''
    data_dir = src_path
    file_list = os.listdir(data_dir)
    data_list = []
    HH_data_list = []
    HV_data_list = []
    VH_data_list = []
    VV_data_list = []
    for f in file_list:
        if f[-1] == "f":
            data_list.append(f)
    for data in data_list:
        if 'HH' in data:
            HH_data_list.append(data)
        elif 'HV' in data:
            HV_data_list.append(data)
        elif 'VH' in data:
            VH_data_list.append(data)
        elif 'VV' in data:
            VV_data_list.append(data)
    merge_data_4band_list = []
    file_name_list = []
    HH_data_list = natsort.natsorted(HH_data_list, alg=natsort.PATH)
    HV_data_list = natsort.natsorted(HV_data_list, alg=natsort.PATH)
    VH_data_list = natsort.natsorted(VH_data_list, alg=natsort.PATH)
    VV_data_list = natsort.natsorted(VV_data_list, alg=natsort.PATH)

    for i in range(len(HH_data_list)):
        imgHH = cv2.imread(osp.join(data_dir, HH_data_list[i]), -1)
        imgHV = cv2.imread(osp.join(data_dir, HV_data_list[i]), -1)
        imgVH = cv2.imread(osp.join(data_dir, VH_data_list[i]), -1)
        imgVV = cv2.imread(osp.join(data_dir, VV_data_list[i]), -1)

        # incase the datavalue equals to 0
        imgHH[imgHH<1] = 1
        imgHV[imgHV<1] = 1
        imgVH[imgVH<1] = 1
        imgVV[imgVV<1] = 1

        imgHH = 10*np.log10(imgHH)
        imgHV = 10*np.log10(imgHV)
        imgVH = 10*np.log10(imgVH)
        imgVV = 10*np.log10(imgVV)

        imgHH = 255*min_max_contrast_median_map(imgHH)
        imgHV = 255*min_max_contrast_median_map(imgHV)
        imgVH = 255*min_max_contrast_median_map(imgVH)
        imgVV = 255*min_max_contrast_median_map(imgVV)

        merge_data_4band = cv2.merge([imgHH, imgHV, imgVV, imgVH])
        merge_data_4band.astype(np.uint8)
        file_name = HH_data_list[i].replace('_HH', '').replace('.tiff', '.png')

        cv2.imwrite(osp.join(out_path, file_name), merge_data_4band)
        print('write ', file_name, ' succeed')


def band3_data_pre_log(path):
    data_dir = path
    file_list = os.listdir(data_dir)
    data_list = []
    HH_data_list = []
    HV_data_list = []
    VV_data_list = []
    for f in file_list:
        if f[-1] == "f":
            data_list.append(f)
    for data in data_list:
        if 'HH' in data:
            HH_data_list.append(data)
        elif 'HV' in data:
            HV_data_list.append(data)
        elif 'VV' in data:
            VV_data_list.append(data)
    merge_data_4band_list = []
    file_name_list = []
    HH_data_list = natsort.natsorted(HH_data_list, alg=natsort.PATH)
    HV_data_list = natsort.natsorted(HV_data_list, alg=natsort.PATH)
    VV_data_list = natsort.natsorted(VV_data_list, alg=natsort.PATH)

    new_dir = osp.join(data_dir, 'v2')
    mylib.mkdir_if_not_exist(new_dir)
    for i in range(len(HH_data_list)):
        imgHH = cv2.imread(osp.join(data_dir, HH_data_list[i]), -1)
        imgHV = cv2.imread(osp.join(data_dir, HV_data_list[i]), -1)
        imgVV = cv2.imread(osp.join(data_dir, VV_data_list[i]), -1)

        # incase the datavalue equals to 0
        imgHH[imgHH<1] = 1
        imgHV[imgHV<1] = 1
        imgVV[imgVV<1] = 1

        imgHH = np.log(imgHH)
        imgHV = np.log(imgHV)
        imgVV = np.log(imgVV)

        imgHH = 255*min_max_map(imgHH)
        imgHV = 255*min_max_map(imgHV)
        imgVV = 255*min_max_map(imgVV)

        merge_data_4band = cv2.merge([imgHH, imgHV, imgVV])
        merge_data_4band.astype(np.uint8)
        file_name = HH_data_list[i].replace('_HH', '').replace('.tiff', '.png')

        cv2.imwrite(osp.join(new_dir, file_name), merge_data_4band)
        print('write ', file_name, ' succeed')


def band4_data_pre_log(path):
    data_dir = path
    file_list = os.listdir(data_dir)
    data_list = []
    HH_data_list = []
    HV_data_list = []
    VV_data_list = []
    VH_data_list = []
    for f in file_list:
        if f[-1] == "f":
            data_list.append(f)
    for data in data_list:
        if 'HH' in data:
            HH_data_list.append(data)
        elif 'HV' in data:
            HV_data_list.append(data)
        elif 'VV' in data:
            VV_data_list.append(data)
        elif 'VH' in data:
            VH_data_list.append(data)
    merge_data_4band_list = []
    file_name_list = []
    HH_data_list = natsort.natsorted(HH_data_list, alg=natsort.PATH)
    HV_data_list = natsort.natsorted(HV_data_list, alg=natsort.PATH)
    VV_data_list = natsort.natsorted(VV_data_list, alg=natsort.PATH)
    VH_data_list = natsort.natsorted(VH_data_list, alg=natsort.PATH)

    new_dir = osp.join(data_dir, 'v2')
    mylib.mkdir_if_not_exist(new_dir)
    for i in range(len(HH_data_list)):
        imgHH = cv2.imread(osp.join(data_dir, HH_data_list[i]), -1)
        imgHV = cv2.imread(osp.join(data_dir, HV_data_list[i]), -1)
        imgVV = cv2.imread(osp.join(data_dir, VV_data_list[i]), -1)
        imgVH = cv2.imread(osp.join(data_dir, VH_data_list[i]), -1)

        # incase the datavalue equals to 0
        imgHH[imgHH<1] = 1
        imgHV[imgHV<1] = 1
        imgVV[imgVV<1] = 1
        imgVH[imgVH<1] = 1

        imgHH = np.log(imgHH)
        imgHV = np.log(imgHV)
        imgVV = np.log(imgVV)
        imgVH = np.log(imgVH)

        imgHH = 255*min_max_map(imgHH)
        imgHV = 255*min_max_map(imgHV)
        imgVV = 255*min_max_map(imgVV)
        imgVH = 255*min_max_map(imgVH)

        merge_data_4band = cv2.merge([imgHH, imgHV, imgVV, imgVH])
        merge_data_4band.astype(np.uint8)
        file_name = HH_data_list[i].replace('_HH', '').replace('.tiff', '.png')

        cv2.imwrite(osp.join(new_dir, file_name), merge_data_4band)
        print('write ', file_name, ' succeed')
        # merge_data_4band_list.append(merge_data_4band)
        # file_name_list.append(file_name)

    # return merge_data_4band_list, file_name_list


if __name__ == "__main__":
    data_path = r"/data/csl/SAR_Seg/SAR_Seg"
    out_path = r'/data/csl/SAR_Seg/SAR_Seg_3/band4/train'
    band4_data_pre_psp(data_path, out_path)
    print('hello')
