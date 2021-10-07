'''
Author: Shuailin Chen
Created Date: 2021-10-07
Last Modified: 2021-10-07
	content: view image and label
'''

import os 
import os.path as osp
import numpy as np
import cv2
import mylib.labelme_utils as lu
import mylib.image_utils as iu
import mylib.file_utils as fu
from PIL import Image
import tifffile
import re

from unsup_seg import read_label_png


if __name__ == '__main__':
    label_dir = r'data/SN6_sup/label_mask'
    img_dir = r'data/SN6_full/SAR-PRO'
    tmp_dir = r'tmp2'
    opacity = 0.5
    orients = fu.read_file_as_list(r'data/SN6_full/SummaryData/SAR_orientations.txt')
    orients = {l.split()[0]: l.split()[1] for l in orients}

    for img_name in os.listdir(img_dir):
        img = tifffile.imread(osp.join(img_dir, img_name))
        # img = cv2.imread(osp.join(img_dir, img_name), -1)

        label_name = img_name.replace('tif', 'png').replace('SAR-Intensity', 'PS-RGB')
        label = read_label_png(osp.join(label_dir, label_name),
                                check_path=False)
        label = (label > 1)*255
        label = np.tile(label[..., None], (1, 1, 3)).astype(np.uint8)
        mixed = (opacity * img + (1-opacity) * label).astype(np.uint8)

        # save images
        mixed_path = osp.join(tmp_dir, 'mixed.png')
        gif_path = osp.join(tmp_dir, 'tt.gif')
        iu.save_image_by_cv2(mixed, mixed_path)
        
        img = Image.fromarray(img)
        label = Image.fromarray(label)
        mixed = Image.fromarray(mixed)
        img.save(gif_path, format='GIF', append_images=[mixed], loop=0, save_all=True, duration=700)

        # print infos
        timestamp = re.findall(r'2019\d{10}_2019\d{10}', img_name)
        assert len(timestamp) == 1
        timestamp = timestamp[0]
        print(f'img: {osp.join(img_dir, img_name)}\nlabel: {osp.join(label_dir, label_name)}\nmixed:{mixed_path}\ngif: {gif_path}\norient: {orients[timestamp]}')
        print()
