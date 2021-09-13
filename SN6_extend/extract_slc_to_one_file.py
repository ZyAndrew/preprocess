'''
Author: Shuailin Chen
Created Date: 2021-09-11
Last Modified: 2021-09-12
	content: extract 4 channel slc data into one file
'''

import os
import os.path as osp
import numpy as np
import cv2
from glob import glob
 
import solaris as sol
import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.label as label
import solaris.preproc.image as image
import solaris.preproc.sar as sar


class GetSLC(pipesegment.PipeSegment):
    '''
    Args:
        output_dir (str): output path of new slc image
        img_dir (str): path to save tmporal images
    '''
    def __init__(self,
                data_root,
                slc_dir,
                output_dir,
                timestamp,
                img_dir,
                ):
                
        super().__init__()

        dst_file_path = osp.join(data_root, output_dir, 
                                    'slc_' + timestamp + '.tif')
        print(f'generating {dst_file_path}')

        # polarimetric calibration
        quads = [
            image.LoadImage(os.path.join(data_root, slc_dir,
                    'CAPELLA_ARL_SM_SLC_'  + pol + '_' + timestamp + '.tif'))
                * sar.CapellaScaleFactor()
            for pol in ['HH', 'HV', 'VH', 'VV']]

        cstack = np.sum(quads) * image.MergeToStack() 

        # orthorectify
        ostack = (cstack
                  * sar.Orthorectify(projection=32631, row_res=.25, col_res=.25)
                  * image.SaveImage(dst_file_path,
                                    no_data_value='nan', return_image=False)          
                 )
        self.feeder = ostack


if __name__ == '__main__':
    path = r'/home/csl/code/preprocess/data/SN6_extend'
    raster_path = osp.join(path, r'SAR-SLC')
    vector_path = osp.join(path, r'geojson_buildings')
    tmp_dir = r'/home/csl/code/preprocess/tmp'

    files = glob(osp.join(raster_path, 'CAPELLA*HH*.tif'))
    print(f'{len(files)} files in total')

    # select the unbroken and unprocessed images 
    timestamps_all = []
    for t in files:
        timestamp = osp.split(t)[1][-33:-4]
        dst_file_path = osp.join(path, 'newSLC2', 
                                    'slc_' + timestamp + '.tif')
        if osp.isfile(dst_file_path):
            print(f'{dst_file_path} already exsits')
            continue
        
        if not (osp.isfile(t) and osp.isfile(t.replace('HH', 'HV')) \
                        and osp.isfile(t.replace('HH', 'VH')) \
                        and osp.isfile(t.replace('HH', 'VV'))):
            print(f'can not fild 4 channel files of {t}')
            continue

        timestamps_all.append(timestamp)

    # timestamps_all = [osp.split(t)[1][-33:-4] for t in files]
    timestamps_unique = sorted(list(set(timestamps_all)))
    arglist = [(path, r'SAR-SLC', 'newSLC2', t, tmp_dir)
                for t in timestamps_unique]
    # GetSLC.parallel(arglist)
    GetSLC.parallel(arglist, processes=2)
