'''
Author: Shuailin Chen
Created Date: 2021-09-11
Last Modified: 2021-09-11
	content: offcial preprocessing script, copied from `https://gist.github.com/dphogan/088c20888bb794f97281d79fd0a36c01`
'''

import os
import glob
import numpy as np

import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.label as label
import solaris.preproc.image as image
import solaris.preproc.sar as sar

class ReprocessSAR(pipesegment.PipeSegment):
    def __init__(self,
                 timestamp='20190804111224_20190804111453',
                 input_dir='/path/to/input/slc',
                 output_dir='/path/to/output/mag_pol'):
        super().__init__()

        # Complex data
        quads = [
            image.LoadImage(os.path.join(input_dir, 'CAPELLA_ARL_SM_SLC_'
                                         + pol + '_' + timestamp + '.tif'))
            * sar.CapellaScaleFactor()
            for pol in ['HH', 'HV', 'VH', 'VV']]
        cstack = np.sum(quads) * image.MergeToStack()

        # Magnitude
        mstack = (cstack
                  * sar.Intensity()
                  * sar.Multilook(5)
                  * sar.Decibels()
                 )

        # Polarimetry
        pstack = (cstack
                  * sar.DecompositionPauli(hh_band=0, vv_band=3, xx_band=1)
                  * image.SelectBands([0,1])
                  * sar.Multilook(5)
                  * sar.Decibels()
                 )

        # All bands, orthorectified
        ostack = ((mstack + pstack)
                  * image.MergeToStack()
                  * sar.Orthorectify(projection=32631, row_res=.25, col_res=.25)
                  * image.SaveImage(os.path.join(output_dir, 'sar_mag_pol_'
                                                 + timestamp + '.tif'),
                                    no_data_value='nan', return_image=False)
                 )
        self.feeder = ostack

files = glob.glob('/path/to/input/slc/*.tif')
timestamps_all = [os.path.split(t)[1][-33:-4] for t in files]
timestamps_unique = sorted(list(set(timestamps_all)))
arglist = [(t,) for t in timestamps_unique]

ReprocessSAR.parallel(arglist)