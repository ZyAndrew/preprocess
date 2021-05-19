'''
Author: Shuailin Chen
Created Date: 2021-05-10
Last Modified: 2021-05-18
	content: extract pauliRGB image from raw SLC file
'''

import os
import os.path as osp
import glob
import numpy as np
import tifffile
import cv2
# import natsort

from mylib import polSAR_utils as psr
from mylib import file_utils as fu

import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.label as label
import solaris.preproc.image as image
import solaris.preproc.sar as sar

class ReadGF3L2(pipesegment.PipeSegment):
    def __init__(self,
                 input_dir='/home/csl/code/preprocess/data/SN6_extend/SAR-SLC',
                 output_dir='/home/csl/code/preprocess/data/SN6_extend/newSLC'):
        super().__init__()

        tifs = glob.glob(osp.join(input_dir, '*.tiff'))
        tifs.sort()

        # Complex data
        # quads = [
        #     image.LoadImage(tif)
        #     * sar.CapellaScaleFactor()
        #     for tif in tifs
        #     ]
        # cstack = np.sum(quads) * image.MergeToStack() 
        cstack = image.LoadImage(tifs[0]) * sar.CapellaScaleFactor()

        # Magnitude
        # mstack = (cstack
        #           * sar.Intensity()
        #         #   * sar.Multilook(3)
        #           * sar.Decibels()
        #          )

        # All bands, orthorectified
        ostack = (cstack
                #   * image.MergeToStack()
                  * image.SaveImage(os.path.join(output_dir, 'intensity.tif'))
                 )
        self.feeder = ostack


folders = [
    r'/home/csl/code/preprocess/data/MGGF3jihuanBC20210427/GF3_KAS_QPSI_024808_W116.2_N37.0_20210427_L2_AHV_L20005617966',
    r'/home/csl/code/preprocess/data/MGGF3jihuanBC20210427/GF3_KAS_QPSI_024808_W116.2_N37.0_20210427_L2_AHV_L20005617966'
]
for folder in folders:
    ReadGF3L2(folder, '/home/csl/code/preprocess/tmp')