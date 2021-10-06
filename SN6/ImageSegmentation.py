'''
Author: Shuailin Chen
Created Date: 2021-09-17
Last Modified: 2021-10-06
	content: adapted from `https://github.com/sirius-mhlee/graph-based-image-segmentation`
'''

import os
import os.path as osp
import cv2
import random as rand
import numpy as np

import GraphOperator as go


def generate_image(ufset, width, height):
    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(width * height)]

    save_img = np.zeros((height, width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            color_idx = ufset.find(y * width + x)
            save_img[y, x] = color[color_idx]

    return save_img


def generate_mask(ufset, width, height):
    save_img = np.zeros((height, width), np.uint8)

    for y in range(height):
        for x in range(width):
            color_idx = ufset.find(y * width + x)
            save_img[y, x] = color_idx

    return save_img


def seg_FH(img, sigma=0.5, k=500, min_size=50, save_path=None):
    ''' main function
    NOTE: it
    
    Args：
        img (ndarray): image with RGB channel order    
    '''
    
    float_img = np.asarray(img, dtype=float)

    gaussian_img = cv2.GaussianBlur(float_img, (5, 5), sigma)
    r, g, b = cv2.split(gaussian_img)
    smooth_img = (r, g, b)

    height, width, channel = img.shape
    graph = go.build_graph(smooth_img, width, height)

    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph, key=weight)

    ufset = go.segment_graph(sorted_graph, width * height, k)
    ufset = go.remove_small_component(ufset, sorted_graph, min_size)
    
    if save_path is not None:
        color_mask = generate_image(ufset, width, height)
        cv2.imwrite(save_path, color_mask)
    save_img = generate_mask(ufset, width, height)
    return save_img


if __name__ == '__main__':

    sigma = 0.5
    k = 500
    min_size = 50
    img_path = r'data/SN6_full/SAR-PRO/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804111224_20190804111453_tile_8683.tif'
    result_path = r'/home/csl/code/preprocess/tmp2/my_result.jpg'
    img = cv2.imread(img_path)
    seg_FH(img, sigma, k, min_size, result_path)
