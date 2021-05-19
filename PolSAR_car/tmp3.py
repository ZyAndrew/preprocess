'''
Author: Shuailin Chen
Created Date: 2021-05-18
Last Modified: 2021-05-18
	content: 
'''
import matplotlib.pyplot as plt
import numpy as np
import os

import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.image as image
import solaris.preproc.sar as sar
import solaris.preproc.optical as optical
import solaris.preproc.label as label

plt.rcParams['figure.figsize'] = [4, 4]
datadir = '../../../solaris/data/preproc_tutorial'

datadir = r'/home/csl/code/preprocess/data/MGGF3jihuanBC20210427/GF3_KAS_QPSI_024808_W116.2_N37.0_20210427_L2_AHV_L20005617966/GF3_KAS_QPSI_024808_W116.2_N37.0_20210427_L2_HH_L20005617966.tiff'

class SARClass(pipesegment.PipeSegment):
    def __init__(self):
        super().__init__()
        self.feeder = (
            # image.LoadImage(os.path.join(datadir, 'sar_hh.tif'))
            image.LoadImage(datadir)
            * sar.CapellaScaleFactor()
            * sar.Intensity() * image.ShowImage(caption='Intensity')
            * sar.Multilook(2) * image.ShowImage(caption='Multilook (Boxcar Filter)')
            * sar.Decibels() * image.ShowImage(caption='Conversion to Decibels')
            # * sar.Orthorectify(projection = 32631, row_res=3, col_res=3) * image.ShowImage(caption='Orthorectification')
            # * image.SaveImage(os.path.join(datadir, 'output3a.tif'))
        )

sar_processing = SARClass()
sar_processing()