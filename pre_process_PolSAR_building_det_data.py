'''
Author: Shuailin Chen
Created Date: 2021-01-25
Last Modified: 2021-03-25
'''
''' 预处理极化SAR建筑物检测数据
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


if __name__=='__main__':
    label_names_all = ('_background_', 'building')
    path = r'data/PolSAR_building_det'
    lu.check_label_name(path, label_names_all)
    lu.json_to_dataset_batch(path, label_names_all)