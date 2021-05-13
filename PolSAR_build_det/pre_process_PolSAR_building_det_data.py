'''
Author: Shuailin Chen
Created Date: 2021-01-25
Last Modified: 2021-05-01
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
import pickle

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

work_dir = './PolSAR_build_det'


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


def flatten_directory(path: str, folder='uni_rot', ori_data_type='mat', select_features=None):
    ''' Flatten hierarchical directory according to label files

    Args:
        path (str): path to label files, not been divided into train, val and 
            test sets
        folder (str): folder in which file data stored, also indicates the 
            data type
        ori_data_type (str): original data type, 'mat': Matlab file, 'npy': 
            numpy file, 'C3' diagonal elements of C3 matrix with logarithm
        select_features (tuple): features to be selected, None indicates to 
            select all. Default: None
    '''

    src_path = path.replace('label', 'data')
    pngs = glob.glob(osp.join(path, '*.png'))
    for png in pngs:
        png = osp.basename(png)
        loc = re.findall(r'[a-z]+', png)[0]
        date = re.findall(r'\d{8}', png)[0]
        idx = re.findall(r'_(\d{3}).', png)[0]
        idx = idx.lstrip('0') if idx.lstrip('0') else '0'

        if folder =='C3':
            src_file_path = osp.join(src_path, loc, date, folder, idx)
        else:
            src_file_path = osp.join(src_path, loc, date, folder, idx, 'unnormed.'+ori_data_type)
        dst_file_path = osp.join(path.replace('label', folder+'_unnormed'), png.replace('png', 'npy'))
        fu.mkdir_if_not_exist(osp.dirname(dst_file_path))
        print(f'copy {src_file_path.replace(src_path, "")}   to   {dst_file_path.replace(src_path, "")}')

        # different original file type
        if ori_data_type == 'mat':
            file = load_uni_rot_mat_file(src_file_path, select_features=select_features)
        elif ori_data_type == 'npy':
            file = np.load(src_file_path)
            # check in nan or inf
            num_nan = np.isnan(file).sum()
            num_inf = np.isinf(file).sum()
            if num_nan > 0:
                raise ValueError(f'{src_file_path}: nan value exist')
            if num_inf > 0:
                raise ValueError(f'{src_file_path}: inf value exist')
        elif ori_data_type == 'C3':
            file = psr.read_c3(src_file_path, out='save_space')
            file = file[(0, 5, 8), :, :]
            file[file<mathlib.eps] = mathlib.eps
            file = np.log(file)
            mathlib.check_inf_nan(file)

        np.save(dst_file_path, file)
    
    print('flatten directory finished\n')


def load_uni_rot_mat_file(path: str, select_features=None) -> ndarray:
    ''' Load uniform rotation matrix file from matlab, also applicable to coherent pattern files
    
    Args:
        path (str): path to the matlab's .mat file
        select_features (tuple): features to be selected, None indicates to 
            select all. Default: None

    Retures:
        uni_rot_sta (ndarray): numpy ndarray stacked by channel 
    '''

    if not path.endswith('unnormed.mat'):
        path = osp.join(path, 'unnormed.mat')
    uni_rot = scipy.io.loadmat(path)
    uni_rot_sta = []

    for k, v in uni_rot.items():
        # skip matlab built-in variables
        if k.endswith('__'):
            continue

        # logarithm for amplitude params
        if ('A_' in k) or ('B_' in k):
            v = np.log(v)
            # print('log for ', k)
        
        # manually select independent features
        if select_features:
            if k in select_features:
                uni_rot_sta.append(v)
                # print(f'select {k} ')
        else:
            uni_rot_sta.append(v)

    # the matlab matrix format should be transpose to numpy matrix format
    assert len(select_features)==len(uni_rot_sta), 'Can\'t find the selected features '
    uni_rot_sta = np.stack(uni_rot_sta, axis=0).transpose(0, 2, 1)

    # check in nan or inf
    num_nan = np.isnan(uni_rot_sta).sum()
    num_inf = np.isinf(uni_rot_sta).sum()
    if num_nan > 0:
        UserWarning(f'{path}: nan value exist')
    if num_inf > 0:
        UserWarning(f'{path}: inf value exist')
        
    return uni_rot_sta


def zscore_uni_rot(path: str, num_channels=7, type='GF3') -> None:
    ''' Zscore the PolSAR rotation matrix, only using statistics from training set, and apply to train, val and test set, also applicable to coherent pattern files
    
    Args:
        path (str): path to the files need to be zscored
        num_channels (float): number of channels. Default: 48
        type (str): 'GF3' or 'RS2'
     '''

    print('zscore process:')
    #  get mean and std value of the training set 
    all_train_files = fu.read_file_as_list(osp.join(work_dir, type+'_training.txt'))

    full_data = np.empty((len(all_train_files), num_channels, 512, 512))
    for ii in tqdm.trange(len(all_train_files)):
        filename = osp.join(path, type, all_train_files[ii]).replace('png', 'npy')
        f = np.load(filename)
        full_data[ii, ...] = f
    
    m = np.mean(full_data, axis=(0, 2, 3), keepdims=True).squeeze(0)
    std = np.std(full_data, axis=(0, 2, 3), ddof=1, keepdims=True).squeeze(0)
    print(f'mean: {m}, \nunbiased std: {std}')
    

    # apply zscore to the training, val, test sets using the derived mean and std value
    print('zscore val files')
    all_val_files = fu.read_file_as_list(osp.join(work_dir, type+'_validation.txt'))
    for ii in tqdm.trange(len(all_val_files)):
        file = all_val_files[ii].replace('png', 'npy')
        src_path = osp.join(path, type, file)
        f = np.load(src_path)
        f = (f-m)/std
        dst_path = osp.join(path, type, 'validation', file).replace('unnormed', 'zscored')
        fu.mkdir_if_not_exist(osp.split(dst_path)[0])
        np.save(dst_path, f)
    
    
    print('zscore test files')
    all_test_files = fu.read_file_as_list(osp.join(work_dir, type+'_test.txt'))
    for ii in tqdm.trange(len(all_test_files)):
        file = all_test_files[ii].replace('png', 'npy')
        src_path = osp.join(path, type, file)
        f = np.load(src_path)
        f = (f-m)/std
        dst_path = osp.join(path, type, 'test', file).replace('unnormed', 'zscored')
        fu.mkdir_if_not_exist(osp.split(dst_path)[0])
        np.save(dst_path, f)
        
    print('zscore train files')
    all_train_files = fu.read_file_as_list(osp.join(work_dir, type+'_training.txt'))
    for ii in tqdm.trange(len(all_train_files)):
        file = all_train_files[ii].replace('png', 'npy')
        src_path = osp.join(path, type, file)
        f = np.load(src_path)
        f = (f-m)/std
        dst_path = osp.join(path, type, 'training', file).replace('unnormed', 'zscored')
        fu.mkdir_if_not_exist(osp.split(dst_path)[0])
        np.save(dst_path, f)
        

def PCA_uni_rot(path, n_components, num_channels=48, save_model=True, save_data=False) -> None:
    ''' Transform zscored uni_rot data using PCA, also applicable to coherent pattern files

    Args:
        path (str): path to zscored data
        n_components (float): param for the PCA function
        num_channels (float): number of channels. Default: 48
        save_model (bool): if save the PCA model. Default: False
        save_data (bool): if save the transformed data. Default: False

    note: only using training set to get the transform matrix, and apply which to the val and test sets
    '''
    
    print('transfrom PolSAR rotation matrix using PCA on:', path)

    # check input
    if osp.isdir(osp.join(path, 'training')):
        train_filenames = os.listdir(osp.join(path, 'training'))
    # elif osp.isdir(osp.join(path, 'train')):
    #     train_filenames = os.listdir(osp.join(path, 'train'))
    else:
        raise IOError('Can not find training set')

    # load training set data
    # train_files = np.expand_dims(np.load(r'data/PolSAR_building_det/zscored/GF3/training/anshou20190223_080.npy'), 0)
    train_files = np.empty((len(train_filenames), num_channels, 512, 512))
    print('loading trainig files')
    for ii in tqdm.trange(len(train_filenames)):
        filename = train_filenames[ii]
        full_file_path = osp.join(path, 'training', filename)
        train_files[ii, ...] = np.load(full_file_path)

    # fit training data to pca model, and save the transformed data to file
    print('fitting PCA model')
    if not isinstance(n_components, (list, tuple)):
        n_components = [n_components]

    for n_component in n_components:
        print(f'\nn_component={n_component}')
        pca = PCA(n_components=n_component)
        train_files = train_files.transpose(0, 2, 3, 1).reshape(-1, num_channels)
        train_files = pca.fit_transform(train_files)
        train_files = train_files.reshape(len(train_filenames), 512, 512, -1)
        print(f'reduced dimension: {train_files.shape}\nvar: {pca.explained_variance_}, \nvar ratio: {pca.explained_variance_ratio_}\n cumsum:\n{np.cumsum(pca.explained_variance_ratio_)}')

        if save_model:
            model_path = osp.split(osp.split(path)[0])[0]
            model_path = osp.join(model_path, f'PCA_{n_component}.p')
            pickle.dump(pca, open(model_path, 'wb'))
            print(f'saved model on {model_path}')
        
        if save_data:
            print('dumping traing data files')
            fu.mkdir_if_not_exist(osp.join(path, 'training').replace('zscored', f'PCA_{n_component}'))
            for ii in tqdm.trange(len(train_filenames)):
                filename = train_filenames[ii]
                full_file_path = osp.join(path, 'training', filename).replace('zscored', f'PCA_{n_component}')
                np.save(full_file_path, train_files[ii, ...])

            del train_files, train_filenames

            # apply model to val set
            val_filenames = os.listdir(osp.join(path, 'validation'))
            val_files = np.empty((len(val_filenames), num_channels, 512, 512))
            print('loading val files')
            for ii in tqdm.trange(len(val_filenames)):
                filename = val_filenames[ii]
                full_file_path = osp.join(path, 'validation', filename)
                val_files[ii, ...] = np.load(full_file_path)

            print('PCA tranfome on val set')
            val_files = val_files.transpose(0, 2, 3, 1).reshape(-1, num_channels)
            val_files = pca.transform(val_files).reshape(len(val_filenames), 512, 512, -1)

            print('dumping val data files')
            fu.mkdir_if_not_exist(osp.join(path, 'validation').replace('zscored', f'PCA_{n_component}'))
            for ii in tqdm.trange(len(val_filenames)):
                filename = val_filenames[ii]
                full_file_path = osp.join(path, 'validation', filename).replace('zscored', f'PCA_{n_component}')
                np.save(full_file_path, val_files[ii, ...])

            del val_files, val_filenames

            # apply model to test set
            test_filenames = os.listdir(osp.join(path, 'test'))
            test_files = np.empty((len(test_filenames), num_channels, 512, 512))
            print('loading test files')
            for ii in tqdm.trange(len(test_filenames)):
                filename = test_filenames[ii]
                full_file_path = osp.join(path, 'test', filename)
                test_files[ii, ...] = np.load(full_file_path)

            print('PCA tranfome on test set')
            test_files = test_files.transpose(0, 2, 3, 1).reshape(-1, num_channels)
            test_files = pca.transform(test_files).reshape(len(test_filenames), 512, 512, -1)

            print('dumping test data files')
            fu.mkdir_if_not_exist(osp.join(path, 'test').replace('zscored', f'PCA_{n_component}'))
            for ii in tqdm.trange(len(test_filenames)):
                filename = test_filenames[ii]
                full_file_path = osp.join(path, 'test', filename).replace('zscored', f'PCA_{n_component}')
                np.save(full_file_path, test_files[ii, ...])

            del test_files, test_filenames


def extract_H_A_alpha_span(path):
    ''' Extract H/A/alpha/Span data 
    Args:
        path (str): to the the H/A/alpha data
    '''

    if osp.isdir(path):
        path = osp.join(path, 'unnormed.npy')
    if not osp.isfile(path):
        raise IOError(f'{path} is not a valid path')


    HAalpha = np.load(path)
    assert HAalpha.shape[0]==3, 'Wrong shape of H/A/ahpha data'

    c3 = psr.read_c3(osp.dirname(path.replace('HAalpha', 'C3')), out='save_space')
    assert np.array_equal(c3.shape[1:], HAalpha.shape[1:]), 'Unmatched C3 and H/A/alpha data pair'

    span = c3[0, ...] + c3[5, ...] + c3[8, ...]
    span[span<mathlib.eps] = mathlib.eps
    span = np.expand_dims(np.log(span), 0)

    HAalphaSpan = np.concatenate((HAalpha, span), axis=0)

    dst_file = path.replace('HAalpha', 'HAalphaSpan')
    print(f'extract H/A/alpha/Span data from {path}   to   {dst_file}')
    fu.mkdir_if_not_exist(osp.dirname(dst_file))
    np.save(dst_file, HAalphaSpan)


if __name__=='__main__':
    ''' extract H/A/alpha/Span data '''
    # path = r'data/PolSAR_building_det/data'
    # for root, dirs, files in os.walk(path):
    #     if root.endswith('HAalpha'):
    #         for fdr in os.listdir(root):
    #             extract_H_A_alpha_span(osp.join(root, fdr))

    ''' flatten_directory '''
    path = r'/data/csl/PolSAR_building_det/label/GF3'
    flatten_directory(path, 
                    folder='cohe_pattern', 
                    ori_data_type='mat', 
                    # select_features=('T23_Cohe_gamma_org', 'C12_Cohe_gamma_org', 'T23_Cohe_gamma_min_min'),
                    select_features=('T23_Cohe_gamma_org', 'C13_Cohe_gamma_contrast', 'T23_Cohe_gamma_contrast'),
                    )

    ''' zscore_uni_rot '''
    path = r'data/PolSAR_building_det/cohe_pattern_unnormed'
    zscore_uni_rot(path, type='GF3', num_channels=3)

    ''' transform standardized uni_rot data using PCA '''
    # path = r'/data/csl/PolSAR_building_det/uni_rot_7_zscored/GF3'
    # # path = r'data/PolSAR_building_det/unnormed/GF3'
    # PCA_uni_rot(path, n_components=(0.99), num_channels=7, save_model=True, save_data=False) 

    ''' split H/A/alpha files '''
    # path = r'data/PolSAR_building_det/data'
    # for root, dirs, files in os.walk(path):
    #     if 'HAalpha' in files:
    #         psr.split_patch_HAalpha(root, patch_size=[512, 512], transpose=False)

                

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