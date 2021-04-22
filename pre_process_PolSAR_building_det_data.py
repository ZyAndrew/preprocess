'''
Author: Shuailin Chen
Created Date: 2021-01-25
Last Modified: 2021-04-22
'''
''' 预处理极化SAR建筑物检测数据
'''
import os
import re
import os.path as osp
import time
import shutil
import os
import os.path as osp
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.decomposition import PCA, IncrementalPCA
from numpy import ndarray
import tqdm

from mylib import polSAR_utils as psr
from mylib import file_utils as fu
from mylib import labelme_utils as lu
from mylib import mathlib

def hoekman_and_norm(src_path:str)->None:
    ''' hoekman decomposition, normalization as pauliRGB, and save to file
    @in     -src_path       -source path, where should contains 'C3' folder
    '''
    if 'C3' in os.listdir(src_path):
        print(f'hoekman and norm on dir: {src_path}', end='')
        c3 = psr.read_c3(osp.join(src_path, 'C3'))
        h = psr.Hokeman_decomposition(c3)

        dst_path = osp.join(src_path, 'Hoekman')
        fu.mkdir_if_not_exist(dst_path)
        # np.save(osp.join(dst_path, 'ori'), h)                 # save the unnormalized file

        for ii in range(9):
            h[ii, :, :] = psr.min_max_contrast_median_map(10*np.log10(h[ii, :, :]))
            # cv2.imwrite(osp.join(dst_path, f'{ii}.jpg'), (h[0, :, :]*255).astype(np.uint8))
            # plt.hist() can take very long time to process a 2D array, but little time to process a 1D array, so flatten the array if possible 
            # plt.hist(h[ii, :, :].flatten())                     
            # plt.savefig(osp.join(dst_path, f'log-hist-{ii}.jpg'))
        np.save(osp.join(dst_path, 'normed'), h)
        print('\tdone')
    else:
        raise ValueError('wrong src path')


def split_hoekman_file(src_path:str, patch_size=(512, 512), filename='normed.npy')->None:
    ''' split hoekman file into patches
    @in     -src_path       -source path, where should contains 'Hoekman' folder
            -patch_size     -size of a patch, in [height, width] format
            -filename       -name of hoekman data file
    '''
    if src_path.split('/')[-1] != 'Hoekman':
        src_path = osp.join(src_path, 'Hoekman')
    print(f'spliting hoekman data on : {src_path}')
    whole_data = np.load(osp.join(src_path, filename))
    whole_het, whole_wes = whole_data.shape[1:]
    idx = 0
    start_x = 0
    start_y = 0
    p_het, p_wes = patch_size
    while start_x<whole_wes and start_y<whole_het:
        # print(f'    spliting the {idx}-th patch')

        # write bin file
        p_data = whole_data[:, start_y:start_y+p_het, start_x:start_x+p_wes]
        p_folder = osp.join(src_path, str(idx))
        fu.mkdir_if_not_exist(p_folder)
        np.save(osp.join(p_folder, 'normed'), p_data)

        # write image, which is cutted from big picture, not re-generated 
        # p_img = (p_data[0, :, :]*255).astype(np.uint8)
        # cv2.imwrite(osp.join(p_folder, 'img.jpg'), p_img)

        # increase patch index
        idx += 1
        start_x += p_wes
        if start_x >= whole_wes:      # next row
            start_x = 0
            start_y += p_het
            if start_y>=whole_het:          # finish
                print('totle split', idx, 'patches done')
                return
            elif start_y+p_het > whole_het: # suplement
                start_y = whole_het - p_het
        elif start_x+p_wes > whole_wes: 
            start_x = whole_wes - p_wes    

    print('all splitted')


def flatten_directory(path: str):
    ''' flatten hierarchical directory '''
    src_path = path.replace('label', 'data')
    pngs = glob.glob(osp.join(path, '*.png'))
    for png in pngs:
        png = osp.basename(png)
        loc = re.findall(r'[a-z]+', png)[0]
        date = re.findall(r'\d{8}', png)[0]
        idx = re.findall(r'_(\d{3}).', png)[0]
        idx = idx.strip('0') if idx.strip('0') else '0'

        src_file_path = osp.join(src_path, loc, date, 'uni_rot', idx, 'unnormed.mat')
        dst_file_path = osp.join(path.replace('label', 'unnormed'), png.replace('png', 'npy'))
        fu.mkdir_if_not_exist(path.replace('label', 'unnormed'))
        print(f'copy {src_file_path} to {dst_file_path}')
        file = load_uni_rot_mat_file(src_file_path)
        np.save(dst_file_path, file)


def load_uni_rot_mat_file(path: str) -> ndarray:
    ''' load uniform rotation matrix file and do a few preprocessing'''
    if not path.endswith('unnormed.mat'):
        path = osp.join(path, 'unnormed.mat')
    uni_rot = scipy.io.loadmat(path)
    uni_rot_sta = []

    for k, v in uni_rot.items():
        # skip matlab built-in variables
        if k.endswith('__'):
            continue

        # logarithm for amplitude params
        if ('A' in k) or ('B' in k):
            v = np.log10(v)
            # print('log for ', k)
        
        uni_rot_sta.append(v)

    uni_rot_sta = np.stack(uni_rot_sta, axis=0)
    return uni_rot_sta


def zscore_uni_rot(path: str) -> None:
    all_files = os.listdir(path)
    full_data = np.empty((len(all_files), 48, 512, 512))
    for ii in tqdm.trange(len(all_files)):
        f = np.load(osp.join(path, all_files[ii]))
        full_data[ii, ...] = f
    
    m = np.mean(full_data, axis=(0, 2, 3))
    std = np.std(full_data, axis=(0, 2, 3), ddof=1)
    print(f'mean: {m}, unbiased std: {std}')
    
    for ii in tqdm.trange(len(all_files)):
        src_path = osp.join(path, all_files[ii])
        f = np.load(src_path)
        dst_path = src_path.replace('unnormed', 'zscored')
        np.save(dst_path, )
        

def standardize_uni_rot(path: str) -> None:
    ''' standardize the uniform PolSAR rotation theory params'''

    # get the mean from the whole dataset
    m = 0
    idx = 0
    for root, dirs, files in os.walk(path):
        if 'unnormed.mat' in files:
            uni_rot_sta = load_uni_rot_mat_file(root)

            idx += 1
            m += ((idx-1)*m + np.mean(uni_rot_sta, axis=(1, 2), keepdims=True))/idx
    print('mean: ', m)

    # variance value
    var = 0
    idx = 0
    for root, dirs, files in os.walk(path):
        if 'unnormed.mat' in files:
            uni_rot_sta = load_uni_rot_mat_file(root)
            idx += 1
            var_now = mathlib.var_with_known_mean(uni_rot_sta, mean=m, axis=(1, 2), ddof=1)
            

    var += np.std(uni_rot_sta, axis=(1, 2), keepdims=True)

    # uni_rot_sta = (uni_rot_sta - m) / std

    # # hist of stardardized data
    # for ii in range(uni_rot_sta.shape[0]):
    #     plt.clf()
    #     plt.hist(uni_rot_sta[ii, :, :].flatten(), bins=100)
    #     plt.savefig(osp.join('./tmp', str(ii)+'.png'))

    # save to file
    np.save(osp.join(path, 'standardized.npy'), uni_rot_sta)


def PCA_uni_rot(path: str) -> None:
    ''' transform standardized uni_rot data using PCA '''
    print('transfrom PolSAR rotation matrix using PCA on:', path)

    for root, dirs, files in os.walk(path):
        if 'standardized.npy' in files:
            uni_rot_sta = np.load(osp.join(root, 'standardized.npy'))

            for ii in range(48):
                std = np.std(uni_rot_sta[ii, :, :])
                print(std)
            pca = PCA(n_components=48)
            pca.fit(uni_rot_sta.reshape(48, -1).transpose())
            print(pca.explained_variance_ratio_)
            print(pca.explained_variance_)


if __name__=='__main__':
    ''' zscore_uni_rot '''
    path = r'data/PolSAR_building_det/unnormed/GF3'
    zscore_uni_rot(path)

    ''' flatten_directory '''
    # path = r'data/PolSAR_building_det/label/RS2'
    # flatten_directory(path)

    ''' standardize uni_rot file '''
    # path = r'data/PolSAR_building_det/data/'
    # standardize_uni_rot(path)

    ''' transform standardized uni_rot data using PCA '''
    # path = r'data/PolSAR_building_det/data/'
    # PCA_uni_rot(path)



    ''' get T3 file '''
    # path = r'data/PolSAR_building_det/data/RS2'
    # for root, dirs, files in os.walk(path):
    #     if osp.split(root)[0].endswith('C3'):
    #         c3 = psr.read_c3(root)
    #         t3 = psr.c32t3(c3=c3)
    #         dst_path = root.replace('C3', 'T3').replace('RS2', 'RS2_copy')
    #         fu.mkdir_if_not_exist(dst_path)
    #         psr.write_t3(dst_path, t3)
    #         print('writted', dst_path)


    # path = r'data/PolSAR_building_det/data/RS2'
    # ''' hoekman decomposition '''
    # for root, dirs, files in os.walk(path):
    #     if 'C3' in dirs:
    #         psr.split_patch(osp.join(root, 'C3'))   # split C3 path
    #         hoekman_and_norm(root)        # hoekman decomposition
    #         split_hoekman_file(root)    # split hoekman decomposition


    # label_names_all = ('_background_', 'building')
    # path = r'data/PolSAR_building_det'
    # lu.check_label_name(path, label_names_all)
    # lu.json_to_dataset_batch(path, label_names_all)

    print('done')