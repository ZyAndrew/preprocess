'''
Author: Shuailin Chen
Created Date: 2021-05-10
Last Modified: 2021-05-20
	content: extract data from capella raw data, perform radiometric 
    correction, WGS 84 projection, and save the pauliRGB images
'''

import os
import glob
import numpy as np
import tifffile

import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.label as label
import solaris.preproc.image as image
import solaris.preproc.sar as sar

class GetPauli(pipesegment.PipeSegment):
    def __init__(self,
                 timestamp='20190804111224_20190804111453',
                 input_dir='/home/csl/code/preprocess/data/SN6_extend/SAR-SLC',
                 output_dir='/home/csl/code/preprocess/data/SN6_extend/pauli'):
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
        pstack = (cstack
                  * sar.DecompositionPauli(hh_band=0, vv_band=3, xx_band=1)
                #   * image.SelectBands([0,1])
                  * sar.Multilook(3)
                  * sar.Decibels()
                 )

        # All bands, orthorectified
        ostack = (pstack
                #   * image.MergeToStack()
                  * sar.Orthorectify(projection=32631, row_res=.25, col_res=.25)
                  * image.SaveImage(os.path.join(output_dir, 'sar_mag_pol_'
                                                 + timestamp + '.tif'),
                                    no_data_value='nan', return_image=False)
                 )
        self.feeder = ostack



# stage 1, get the pauli decomposition, save to 'data/SN6_extend/pauli'
files = glob.glob('/home/csl/code/preprocess/data/SN6_extend/SAR-SLC/*.tif')
timestamps_all = [os.path.split(t)[1][-33:-4] for t in files]
timestamps_unique = sorted(list(set(timestamps_all)))
arglist = [(t,) for t in timestamps_unique]
GetPauli.parallel(arglist, processes=1)


# stage 2, auto-contrast the pauliRGB, save to 'data/SN6_extend/pauli2'
files = glob.glob(r'/home/csl/code/preprocess/data/SN6_extend/pauli/*.tif')
for file in files:
  im = tifffile.imread(file)
  max_ = np.nanmax(im, axis=(0, 1), keepdims=True)
  min_ = np.nanmin(im, axis=(0, 1), keepdims=True)
  im = (im-min_) / (max_-min_)
  im *= 255
  im = im.astype(np.uint8)
  dst_file = file.replace('pauli', 'pauli2')
  tifffile.imsave(dst_file, im)
  print(f'save {dst_file}')
print('done')