'''
Author: Shuailin Chen
Created Date: 2021-09-16
Last Modified: 2021-09-16
	content: convert original label to meet the requeirements of mmseg，set the mask to 0 if the RGB pixel is invalid，set to 1 if is backgound, 2 if is building
'''

import os
import os.path as osp
import cv2
import mylib.labelme_utils as lu
import numpy as np
import tifffile
from PIL import Image


def convert_labels(src_folder, dst_folder, SAR_folder, tmp_dir, colormap):

    print(f'totally {len(os.listdir(src_folder))} mask images')
    for idx, mask in enumerate(os.listdir(src_folder)):
        old_mask_path = osp.join(src_folder, mask)
        SAR_path = old_mask_path.replace('PS-RGB', 'SAR-Intensity') \
                                .replace(src_folder, SAR_folder)

        # binarize mask
        ori_mask = cv2.imread(old_mask_path, -1)
        assert ori_mask.ndim == 2, \
                    f'expect ndim of mask image is 2, got {ori_mask.ndim}'
        new_mask = ori_mask > 0
        new_mask_path = osp.join(dst_folder, mask).replace('tif', 'png')

        # set invalid pixel value in mask to 0, background pixel value to 1, while building to 2
        new_mask = new_mask.astype(np.uint8) + 1
        SAR_img = cv2.cvtColor(cv2.imread(SAR_path, -1), cv2.COLOR_BGR2GRAY)
        closed = (SAR_img > 0).astype(np.uint8)
        h, w = closed.shape
        # 扩展这个mask，因为处于图像边缘的0像素，经过闭运算后会变成1，因此将其边界扩展以避免这样的情况
        tmp = np.zeros((h+100, w+100))
        tmp[50: h+50, 50:w+50] = closed
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)
        closed = tmp[50: h+50, 50:w+50]
        closed_path = osp.join(tmp_dir, mask.replace('tif', 'gif'))
        SAR_img = Image.fromarray(SAR_img)
        new_mask[closed==0] = 0    
        SAR_img.save(closed_path, format='GIF', append_images=[Image.fromarray(closed*255), Image.fromarray(new_mask*127)], loop=0, save_all=True, duration=700)    

        # save
        assert new_mask.max()<=2
        print(f'SAR path: {SAR_path}')
        print(f'closed path: {closed_path}')
        print(f'{idx}: saving {new_mask_path}')
        lu.lblsave(new_mask_path, new_mask, colormap=colormap)
        pass


if __name__ == '__main__':
    src_folder = r'/home/csl/code/preprocess/data/SN6_full/mask'
    SAR_folder = r'/home/csl/code/preprocess/data/SN6_full/SAR-PRO'
    dst_folder = r'/home/csl/code/preprocess/data/SN6_full_mask'
    tmp_dir = r'/home/csl/code/preprocess/tmp'
    PALETTE = np.array([[128, 128, 128], [0, 0, 0], [255, 255, 255]])
    convert_labels(src_folder, dst_folder, SAR_folder, tmp_dir, PALETTE)
