'''
Author: Shuailin Chen
Created Date: 2021-09-11
Last Modified: 2021-09-11
	content: 
'''

import os
import os.path as osp
import numpy as np
import cv2
import solaris as sol
 
 




if __name__ == '__main__':
    path = r'/home/csl/code/preprocess/data/SN6_extend'
    raster_path = osp.join(path, r'SAR-SLC')
    vector_path = osp.join(path, r'geojson_buildings')

    raster_tiler = sol.tile.raster_tile.RasterTiler(
        dest_dir=osp.join(path, r'tile'),  # the directory to save images to
        # QUERY: why this is src?
        src_tile_size=(512, 512),   # the size of the output chips
        verbose=True)

    raster_bounds_crs = raster_tiler.tile(osp.join(raster_path,
        r'CAPELLA_ARL_SM_SLC_HH_20190804111224_20190804111453.tif'))
