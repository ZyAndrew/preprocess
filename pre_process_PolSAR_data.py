'''
Author: Shuailin Chen
Created Date: 2021-01-25
Last Modified: 2021-04-19
'''
''' 预处理极化SAR数据
    1) 将一整张图分成多个小 patch 
    2) 做 Hoekman 分解，并分成多个小 patch
'''
import os
import re
import os.path as osp
import time
import shutil
from mylib import polSAR_utils as psr
from mylib import file_utils as fu
from mylib import labelme_utils as lu
import cv2
import numpy as np
import matplotlib.pyplot as plt

def hoekman_and_norm(src_path:str)->None:
    ''' hoekman decomposition, normalization as pauliRGB, and save to file, the new version is hoekman() func.
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


def hoekman(src_path:str, norm=False)->None:
    ''' Hoekman decomposition and save to file, the hoekman_and_norm() is an older version of this func, and is a subset of this func
    @in     -src_path       -source path, where should contains 'C3' folder
    @in     -norm           -normalize or not, default: false
    '''
    if 'C3' in os.listdir(src_path):
        # read C3 file
        print(f'hoekman on dir (norm={norm}): {src_path}', end='')
        c3 = psr.read_c3(osp.join(src_path, 'C3'))
        h = psr.Hokeman_decomposition(c3)

        dst_path = osp.join(src_path, 'Hoekman')
        fu.mkdir_if_not_exist(dst_path)
        # np.save(osp.join(dst_path, 'ori'), h)                 # save the unnormalized file

        # normalize
        if norm:
            for ii in range(9):
                h[ii, :, :] = psr.min_max_contrast_median_map(10*np.log10(h[ii, :, :]))
                # cv2.imwrite(osp.join(dst_path, f'{ii}.jpg'), (h[0, :, :]*255).astype(np.uint8))
                # plt.hist() can take very long time to process a 2D array, but little time to process a 1D array, so flatten the array if possible 
                # plt.hist(h[ii, :, :].flatten())                     
                # plt.savefig(osp.join(dst_path, f'log-hist-{ii}.jpg'))

        # save to file
        if norm:
            np.save(osp.join(dst_path, 'normed'), h)
        else:
            np.save(osp.join(dst_path, 'unnormed'), h)
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
        np.save(osp.join(p_folder, filename), p_data)

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



def get_data_mean_std(path, data_type='c3'):
    ''' deprecated  
    get mean and std value of a whole dataset of PolSAR data,
        assume that the data distribution is even among all the PolSAR files, which is actually not
    '''

    # first, get mean value 
    print('calculating mean value')
    mean = np.zeros(6, dtype=np.complex64)
    num_caled = 0
    for root, dirs, files in os.walk(path):
        if 'C3' == root[-2:]:
            c3 = psr.read_c3(root, out='complex_vector_6', is_print=True)
            num_now = c3.shape[1]*c3.shape[2]
            mean_now = c3.mean(axis=(-1,-2))
            mean = num_caled/(num_caled+num_now)*mean + num_now/(num_now+num_caled)*mean_now
            num_caled += num_now
            print(f'mean of whole: {mean}\nmean of now: {mean_now}')

    # second get std value
    print('calculating stc value')
    var = np.zeros(6)       #should be a real number
    num_caled = 0
    for root, _, _ in os.walk(path):
        if 'C3' in root:
            c3 = psr.read_c3(root, out='complex_vector_6', is_print=True)
            num_now = c3.shape[1]*c3.shape[2]
            var_now = np.sum(np.abs(c3-mean)**2)/num_now
            var = num_caled/(num_caled+num_now)*var + num_now/(num_caled+num_now)*var_now
            num_caled += num_now
    std = np.sqrt(var)
    return mean, std
            

def check_data_value_scale(path):
    ''' check the scale of PolSAR data value '''
    for root, _, _ in os.walk(path):
        if 'C3' == root[-2:]:
            c3 = psr.read_c3(root, out='save_space')
            if c3.mean()>0:
                print(f'{root} : mean value is {c3.mean()}')

 
def split_train_val_test_SAR_CD(path):
    ''' Split train, val and test set randomly for SAR_CD dataset,
    regardless of its orbit direction and sensing time    
    '''
    for root, dirs, files in os.walk(path):
        if re.findall(r'[\x80-\xff]{4}')




if __name__=='__main__':
    ''' hoekman decomposition '''
    path = r'data/SAR_CD/RS2/data'
    for root, dirs, files in os.walk(path):
        if 'C3' in dirs:
            # print(root)
            hoekman(root, norm=False)
            split_hoekman_file(root, filename='unnormed.npy')


    ''' write mean and std value of the big picture '''
    # path = r'/data/csl/SAR_CD/GF3/data/'
    # for root, _, _ in os.walk(path):
    #     if 's2' in root[-2:]:
    #         s2 = psr.read_s2(root, is_print=True)
    #         mean, std, _ = psr.norm_3_sigma(s2, type='abs')
    #         np.save(osp.join(root, 's2_abs_mean'), mean)
    #         np.save(osp.join(root, 's2_abs_std'), std)


    ''' test psr.norm_3_sigma() func '''
    # a = [1+1j, -1-1j, 18+1j, -18-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+17j, -1-17j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j, -1-1j, 1+1j]
    # b = np.array(a).reshape(2, 3, 7)
    # c = psr.norm_3_sigma(b)
    # print(c)


    # path = r'/data/csl/SAR_CD/RS2/data/'
    # check_data_value_scale(path)

    # ''' get statistics of PolSAR raw data '''
    # path = r'/data/csl/SAR_CD/GF3/data/'
    # mean, std = get_data_mean_std(path)




    ''' hoekman decomposition '''
    # # path=r'/home/csl/code/preprocess/data/SAR_CD/GF3/data/E132_N34_日本安芸/降轨/1/20170531/C3'
    # # psr.split_patch(path, transpose=True)

    # path = r'/data/csl/SAR_CD/RS2/data/'
    # for root, dirs, files in os.walk(path):
    #     if 'C3' in dirs:
    #         # psr.split_patch(osp.join(root, 'C3'), transpose=True)
    #         hoekman_and_norm(root)
    #         split_hoekman_file(root)





    # path = r'/home/csl/code/preprocess/data/SAR_CD/GF3/data'
    # for root, dirs, files in os.walk(path):
    #     if 'C3' in dirs:
    #         details = root.split('/')[-1]
    #         tm = re.search('\d{8}', details).group()
    #         with open(osp.join(root, 'README.txt'), 'w') as f:
    #             f.write(f'详细的产品信息为: {details}, 文件夹名字仅包含了时间信息')
    #         print(f'rename {details} to {tm}', end='')
    #         os.rename(root, root.replace(details, tm))
    #         print('\tdone')

    print('all done')