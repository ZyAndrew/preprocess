'''
Author: Shuailin Chen
Created Date: 2021-04-25
Last Modified: 2021-05-01
	content: 
'''


import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import scipy.io
from mpl_toolkits import mplot3d
import cv2
# from PolSAR_build_det import pre_process_PolSAR_building_det_data as pp
import pre_process_PolSAR_building_det_data as pp
# from PolSAR_build_det.pre_process_PolSAR_building_det_data import load_uni_rot_mat_file


# # path = r'/home/csl/code/preprocess/data/PolSAR_building_det/uni_rot_zscored/GF3/training/anshou20190223_080.npy'
# path = r'data/PolSAR_building_det/data/GF3/hebei/20180719/uni_rot/50'
# # pp.load_uni_rot_mat_file(path)

# data = scipy.io.loadmat(osp.join(path, 'unnormed.mat'))

# x = y = np.arange(0, 512, 1)
# X, Y = np.meshgrid(x, y)

# uni_rot_sta = []
# for ii, (k, v) in enumerate(data.items()):
#     # skip matlab built-in variables
#     if k.endswith('__'):
#         print(f'{ii}: continue')
#         continue

#     if ('A_' in k) or ('B_' in k):
#         v = np.log(v)
#         print(f'{ii}: log for {k}')

#     # uni_rot_sta.append(v)

# # uni_rot_sta = np.stack(uni_rot_sta, axis=0)
# # np.save('./tmp3/a.npy', uni_rot_sta)

#     # check for inf and nan
#     # num_nan = np.isnan(v).sum()
#     # num_inf = np.isinf(v).sum()
#     # print(f'{ii}, {k}: #nan: {num_nan}, #inf: {num_inf}')
#     # if (num_inf>0) or (num_nan>0):
#     #     Warning('nan or inf')

#     # display
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.plot_surface(X, Y, v)
#     # plt.savefig(osp.join(work_dir, f'{ii}_{k}_3d.png'))
#     # plt.clf()

#     # display
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(v)
#     fig.colorbar(cax)
#     plt.savefig(osp.join(work_dir, f'{ii}_{k}.png'))
#     plt.clf()


work_dir=  r'/home/csl/code/preprocess/tmp'
# path = r'/home/csl/code/preprocess/data/PolSAR_building_det/uni_rot_zscored/GF3/training/anshou20190223_080.npy'
path = r'data/PolSAR_building_det/cohe_pattern_zscored/GF3/validation/hebei20190505_184.npy'
# path = r'./tmp3/a.npy'


data = np.load(path)

# # data[:8, ...] = np.exp(data[:8, ...])
# diff = np.abs(data - uni_rot_sta)
# print(f'diff sum: {diff.sum()}, max: {diff.max()}')

print(data.shape)
for ii in range(data.shape[0]):
    fig = plt.figure()
    
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, data[ii, ...])
    # plt.savefig(osp.join(work_dir, f'{ii}_{k}_3d.png'))
    # plt.clf()

    ax = fig.add_subplot(111)
    cax = ax.matshow(data[ii, ...], cmap=plt.get_cmap('jet'))
    fig.colorbar(cax)
    # plt.hist(data[ii, ...].flatten())
    # plt.title(str(ii))
    plt.savefig(osp.join(work_dir, f'{str(ii)}.png'))
    plt.clf()




# work_dir=  r'/home/csl/code/preprocess/tmp2'
# # path = r'/home/csl/code/preprocess/data/PolSAR_building_det/uni_rot_zscored/GF3/training/anshou20190223_080.npy'
# path = r'data/PolSAR_building_det/uni_rot_zscored/GF3/validation/hengbin20190615_066.npy'
# # path = r'./tmp3/a.npy'


# data = np.load(path)

# # # data[:8, ...] = np.exp(data[:8, ...])
# # diff = np.abs(data - uni_rot_sta)
# # print(f'diff sum: {diff.sum()}, max: {diff.max()}')

# print(data.shape)
# for ii in range(data.shape[0]):
#     fig = plt.figure()
    
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.plot_surface(X, Y, data[ii, ...])
#     # plt.savefig(osp.join(work_dir, f'{ii}_{k}_3d.png'))
#     # plt.clf()

#     ax = fig.add_subplot(111)
#     cax = ax.matshow(data[ii, ...])
#     fig.colorbar(cax)
#     # plt.hist(data[ii, ...].flatten())
#     # plt.title(str(ii))
#     plt.savefig(osp.join(work_dir, f'{str(ii)}.png'))
#     plt.clf()

