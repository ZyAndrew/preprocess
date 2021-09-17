'''
Author: Shuailin Chen
Created Date: 2021-09-13
Last Modified: 2021-09-16
	content: try to convert the valid mask of PS-RGB to building annotations, failed, it seems like the valid mask of PS-RGB is errorneous, which including the whole image
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.image as image
import solaris.preproc.sar as sar
import solaris.preproc.optical as optical
import solaris.preproc.label as label


class TransferMask(pipesegment.PipeSegment):
    def __init__(self, masked_path, unmasked_path, output_path, tmp_dir):
        super().__init__()
        load_masked = image.LoadImage(masked_path) \
                    * image.SaveImage(osp.join(tmp_dir, 'rgb.tif'))
        load_unmasked = image.LoadImage(unmasked_path) \
                    * image.SaveImage(osp.join(tmp_dir, 'mask.png'))
        get_mask = image.GetMask()
        set_mask = image.SetMask(0)
        save_output = image.SaveImage(output_path)
        self.feeder = (load_unmasked + load_masked * get_mask) * set_mask * save_output


tmp_dir = r'/home/csl/code/preprocess/tmp'
masked_path = r'/home/csl/code/preprocess/data/SN6_extend/tile_rgb/900/SN6_AOI_11_Rotterdam_PS-RGB_TrainVal_601261_5753558.tif'
unmasked_path = r'/home/csl/code/preprocess/data/SN6_extend/tile_mask/900/geoms_601261_5753558.png'
output_path = os.path.join(tmp_dir, 'masked.tif')
transfer_mask = TransferMask(masked_path, unmasked_path, output_path, tmp_dir)
transfer_mask()


load_masked = (image.LoadImage(masked_path)() \
            * image.SaveImage(osp.join(tmp_dir, 'rgb.tif')))()
load_unmasked = (image.LoadImage(unmasked_path)() \
            * image.SaveImage(osp.join(tmp_dir, 'mask.tif')))()

get_mask = (load_masked * image.GetMask())()

set_mask = image.SetMask(0)

new = ()
save_output = image.SaveImage(output_path)