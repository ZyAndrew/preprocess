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

from mylib import polSAR_utils as psr
from mylib import file_utils as fu

import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.label as label
import solaris.preproc.image as image
import solaris.preproc.sar as sar

class GetSLC(pipesegment.PipeSegment):
    def __init__(self,
                 timestamp='20190804111224_20190804111453',
                 input_dir='/home/csl/code/preprocess/data/SN6_extend/SAR-SLC',
                 output_dir='/home/csl/code/preprocess/data/SN6_extend/newSLC'):
        super().__init__()

        # Complex data
        quads = [
            image.LoadImage(os.path.join(input_dir, 'CAPELLA_ARL_SM_SLC_'
                                         + pol + '_' + timestamp + '.tif'))
            * sar.CapellaScaleFactor()
            for pol in ['HH', 'HV', 'VH', 'VV']]
        cstack = np.sum(quads) * image.MergeToStack() 
        # cstack  sar.MultilookComplex(3)

        # Magnitude
        # mstack = (cstack
        #           * sar.Intensity()
        #           * sar.Multilook(5)
        #           * sar.Decibels()
        #          )

        # Polarimetry
        # pstack = (cstack
        #           * sar.DecompositionPauli(hh_band=0, vv_band=3, xx_band=1)
        #         #   * image.SelectBands([0,1])
        #           * sar.Multilook(3)
        #           * sar.Decibels()
        #          )

        # All bands, orthorectified
        ostack = (cstack
                #   * image.MergeToStack()
                  * sar.Orthorectify(projection=32631, row_res=.25, col_res=.25)
                  * image.SaveImage(os.path.join(output_dir, 'sar_mag_pol_'
                                                 + timestamp + '.tif'),
                                    no_data_value='nan', return_image=False)
                 )
        self.feeder = ostack



# stage 1, get the pauli decomposition
# files = glob.glob('/home/csl/code/preprocess/data/SN6_extend/SAR-SLC/CAPELLA_ARL_SM_SLC_HH_20190822093113_20190822093410.tif')
# timestamps_all = [os.path.split(t)[1][-33:-4] for t in files]
# timestamps_unique = sorted(list(set(timestamps_all)))
# arglist = [(t,) for t in timestamps_unique]
# GetSLC.parallel(arglist, processes=1)

# stage 2, extract the ROI area from SN6
# timestamps = ('20190822093113_20190822093410', '20190822072404_20190822072642')
# # in the format of (x, y, w, h), where x and y are the coordinates the lower right corner
# rois = [[[28455, 926, 1151, 554], [27472, 1552, 718, 468]], [[23146, 1748, 977, 342]]]
# src_path = r'/home/csl/code/preprocess/data/SN6_extend/newSLC'
# dst_path = r'/home/csl/code/preprocess/data/SN6_extend/newSLC/ROI'
# img_prefix = r'sar_mag_pol_'
# for timestamp, rois_ in zip(timestamps, rois):
#     # read tif
#     img_path = osp.join(src_path, img_prefix+timestamp+'.tif')
#     img = tifffile.imread(img_path)
#     print(f'\npath: {img_path}\nshape: {img.shape}\ndtype: {img.dtype}\n{"-"*100}')

#     # generate pauliRGB
#     pauli = psr.rgb_by_s2(img.transpose(2, 0, 1))
#     pauli = (pauli*255).astype(np.uint8)

#     # extract roi 
#     for ii, roi in enumerate(rois_):
#         # mkdir
#         dst_folder = osp.join(dst_path, timestamp, str(ii))
#         fu.mkdir_if_not_exist(dst_folder)

#         with open(osp.join(dst_folder, 'README.txt'), 'w') as f:
#             f.write(f'ROI: {roi}\nin the format of (x, y, w, h), where x and y are the coordinates the lower right corner')
        
#         xs = roi[0] - roi[2]+1
#         ys = roi[1] - roi[3]+1
#         xe = roi[0] + 1
#         ye = roi[1] + 1
#         img_roi = img[ys:ye, xs:xe, ...]
#         psr.write_s2(dst_folder, img_roi.transpose(2, 0, 1))
        
#         pauli_roi = pauli[ys:ye, xs:xe, ...]
#         cv2.imwrite(osp.join(dst_folder, 'pauli.png'), pauli_roi)

# stage 3, extract ROI area from GF3 data
dst_folder = r'/home/csl/code/preprocess/PolSAR_car/ship'
paren_folder = r'/home/csl/code/preprocess/data/SAR_CD/GF3/data'
locations = [r'E130_N34_日本鞍手/降轨/1', r'E139_N35_日本横滨/降轨/1']
times = [[20190602, 20170607, 20190602], [20190615]]
rois = [[[3660, 2491, 49, 74], [3743, 2426, 69, 66], [3812, 2449, 62, 58]], [[1436, 807, 185, 137]]]
for loc, times_, rois_ in zip(locations, times, rois):
    for tm, roi in zip(times_, rois_):
        src_path = osp.join(paren_folder, loc, str(tm), r'C3')
        dst_path = osp.join(dst_folder, loc, str(tm), r'C3')
        if osp.exists(dst_path):
            dst_path = osp.join(osp.split(dst_path)[0], r'2', r'C3')
        psr.exact_patch_C3(src_path, roi, dst_path)
