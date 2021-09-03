'''
Author: Shuailin Chen
Created Date: 2021-06-14
Last Modified: 2021-06-19
	content: calculate statistics of the rsaicp datasets
'''
import cv2
import os
import os.path as osp
import numpy as np


if __name__ == '__main__':
    # original
    # path = r'/home/csl/code/preprocess/data/S2Looking'
    # statistics = {}
    # for split in os.listdir(path):
    #     print(f'{split}:')
    #     for category in os.listdir(osp.join(path, split)):
    #         print(f'  {category}:')
    #         imgs = []
    #         img_cnt = 0
    #         for img in os.listdir(osp.join(path, split, category)):
    #             img_cnt += 1
    #             imgs.append(cv2.imread(osp.join(path, split, category, img), -1))

    #         imgs = np.stack(imgs, axis=0)
    #         print(f'    #samples: {img_cnt}')

    #         if 'Image' in category:
    #             mean = imgs.mean(axis=(0, 1, 2))
    #             std = imgs.std(axis=(0, 1, 2))
    #             print(f'    mean: {mean}, std: {std}')
    #         elif 'label' in category:
    #             CD_cnt = (imgs>0).sum()
    #             print(f'    CD_cnt: {CD_cnt}')
    #         else:
    #             raise ValueError

    
    # add label folder 
    path = r'/home/csl/code/preprocess/data/S2Looking'
    statistics = {}
    for split in os.listdir(path):
        if not osp.isdir(osp.join(path, split)):
            continue
        print(f'{split}:')
        category = 'label'
        print(f'  {category}:')

        imgs = []
        img_cnt = 0
        for img in os.listdir(osp.join(path, split, category)):
            img_cnt += 1
            imgs.append(cv2.imread(osp.join(path, split, category, img), -1))

        imgs = np.stack(imgs, axis=0)
        print(f'    #samples: {img_cnt}')

        cd_map = imgs > 0
        cd_map = cd_map.astype(np.int32)

        for ii in range(3):
            cd_map[..., ii] *= (ii+1)
        
        cd_map_ensemble = cd_map.sum(axis=-1)
        value, counts = np.unique(cd_map_ensemble, return_counts=True)
        assert counts.sum() == 1024**2 * img_cnt
        percents = counts / (1024**2 * img_cnt)

        # 0: no change
        # 1: building appear
        # 3: buildign disappear
        # 4: building appear and disappear
        # others: illegal
        print(f'    value: {value}, counts: {counts}, percents: {percents}')
                
