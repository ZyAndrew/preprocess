'''
Author: Shuailin Chen
Created Date: 2021-05-10
Last Modified: 2021-06-20
  content: extract data from capella raw data, perform radiometric 
    correction, WGS 84 projection, and save SLC data, and extract the ROI area of SN6 and GF3 data
'''

import os
import os.path as osp
import glob
import numpy as np
import tifffile
import cv2

from mylib import polSAR_utils as psr
from mylib import file_utils as fu
from mylib import image_utils as iu

# import solaris.preproc.pipesegment as pipesegment
# import solaris.preproc.label as label
# import solaris.preproc.image as image
# import solaris.preproc.sar as sar

# class GetSLC(pipesegment.PipeSegment):
#     def __init__(self,
#                  timestamp='20190804111224_20190804111453',
#                  input_dir='/home/csl/code/preprocess/data/SN6_extend/SAR-SLC',
#                  output_dir='/home/csl/code/preprocess/data/SN6_extend/newSLC'):
#         super().__init__()

#         # Complex data
#         quads = [
#             image.LoadImage(os.path.join(input_dir, 'CAPELLA_ARL_SM_SLC_'
#                                          + pol + '_' + timestamp + '.tif'))
#             * sar.CapellaScaleFactor()
#             for pol in ['HH', 'HV', 'VH', 'VV']]
#         cstack = np.sum(quads) * image.MergeToStack() 
#         # cstack  sar.MultilookComplex(3)

#         # Magnitude
#         # mstack = (cstack
#         #           * sar.Intensity()
#         #           * sar.Multilook(5)
#         #           * sar.Decibels()
#         #          )

#         # Polarimetry
#         # pstack = (cstack
#         #           * sar.DecompositionPauli(hh_band=0, vv_band=3, xx_band=1)
#         #         #   * image.SelectBands([0,1])
#         #           * sar.Multilook(3)
#         #           * sar.Decibels()
#         #          )

#         # All bands, orthorectified
#         ostack = (cstack
#                 #   * image.MergeToStack()
#                   * sar.Orthorectify(projection=32631, row_res=.25, col_res=.25)
#                   * image.SaveImage(os.path.join(output_dir, 'sar_mag_pol_'
#                                                  + timestamp + '.tif'),
#                                     no_data_value='nan', return_image=False)
#                  )
#         self.feeder = ostack



''' stage 1, save the preprocessed SLC data'''
# files = glob.glob('/home/csl/code/preprocess/data/SN6_extend/SAR-SLC/CAPELLA_ARL_SM_SLC_HH_20190822093113_20190822093410.tif')
# timestamps_all = [os.path.split(t)[1][-33:-4] for t in files]
# timestamps_unique = sorted(list(set(timestamps_all)))
# arglist = [(t,) for t in timestamps_unique]
# GetSLC.parallel(arglist, processes=1)


''' stage 2, extract the ROI area from SN6'''
# # extract s2 data
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

#     # extract ROI 
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


''' stage 3, extract ROI area (car and ship) from GF3 data'''
# dst_folder = r'/home/csl/code/preprocess/PolSAR_car/data/ship'
# paren_folder = r'/home/csl/code/preprocess/data/SAR_CD/GF3/data'
# locations = [r'E130_N34_日本鞍手/降轨/1', r'E139_N35_日本横滨/降轨/1']
# times = [[20190602, 20170607, 20190602], [20190615]]
# rois = [[[3660, 2491, 49, 74], [3743, 2426, 69, 66], [3812, 2449, 62, 58]], [[1436, 807, 185, 137]]]
# for loc, times_, rois_ in zip(locations, times, rois):
#     for tm, roi in zip(times_, rois_):
#         src_path = osp.join(paren_folder, loc, str(tm), r'C3')
#         dst_path = osp.join(dst_folder, loc, str(tm), r'C3')
#         if osp.exists(dst_path):
#             dst_path = osp.join(osp.split(dst_path)[0], r'2', r'C3')
#         psr.exact_patch_C3(src_path, roi, dst_path)


''' stage 4, extract ROI area (building) from GF3 data'''
# dst_folder = r'/home/csl/code/preprocess/PolSAR_car/data/building'
# paren_folder = r'/home/csl/code/preprocess/data/SAR_CD/GF3/data'
# locations = [r'E139_N35_日本横滨/降轨/1']
# times = [[20190615]]
# rois = [[[1337, 1721, 354, 316]]]
# for loc, times_, rois_ in zip(locations, times, rois):
#     for tm, roi in zip(times_, rois_):
#         src_path = osp.join(paren_folder, loc, str(tm), r'C3')
#         dst_path = osp.join(dst_folder, loc, str(tm), r'C3')
#         if osp.exists(dst_path):
#             dst_path = osp.join(osp.split(dst_path)[0], r'2', r'C3')
#         psr.exact_patch_C3(src_path, roi, dst_path)



''' stage 5, extract supplementary ROI area (building) from GF3 data'''
dst_folder = r'/home/csl/code/preprocess/PolSAR_car/data/building'
parent_folder = r'/home/csl/code/preprocess/data/云南地震配准GF3数据'
products = [r'GF3_SAY_QPSI_010060_E99.9_N25.6_20180708_L1A_AHV_L10003309961',
            r'GF3_KAS_QPSI_025232_E99.8_N25.7_20210526_L1A_AHV_L10005666941'
            ]
rois = [[6824, 2961, 227, 238], [6802, 321, 128, 130]]
for product, roi in zip(products, rois):
    src_path = osp.join(parent_folder, product)
    c3_path = osp.join(src_path, r'C3')
    # s2 = psr.read_s2(src_path, is_print=True)
    # pauli = psr.rgb_by_s2(s2, if_mask=True)
    # iu.save_image_by_cv2(pauli, dst_path=osp.join(src_path, 'pauli_s2.jpg'))
    
    # # c3 = psr.s22c3(s2=s2)
    # c3 = psr.read_c3(c3_path, is_print=True)

    # t3 = psr.c32t3(c3=c3)
    # pauli = psr.rgb_by_t3(t3, if_mask=True)
    # iu.save_image_by_cv2(pauli, osp.join(src_path, 'pauli_t3.jpg'))
    
    # pauli = psr.rgb_by_c3(c3, if_mask=True)
    # iu.save_image_by_cv2(pauli, dst_path=osp.join(c3_path, 'pauli.jpg'))
    # psr.write_c3(c3_path, c3, is_print=True)

    patch_path = osp.join(src_path, r'patch')
    patch_c3_path = osp.join(patch_path, r'C3')
    psr.exact_patch_s2(src_path, roi, patch_path, if_mask=True)
    psr.exact_patch_C3(c3_path, roi, patch_c3_path, if_mask=True)
