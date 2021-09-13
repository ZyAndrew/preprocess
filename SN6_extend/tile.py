'''
Author: Shuailin Chen
Created Date: 2021-09-12
Last Modified: 2021-09-13
	content: tile the image into patches according the PS-RGB image
'''

import os
import os.path as osp
import numpy as np
import cv2
from glob import glob
from mylib.types import LINE_SEPARATOR
import re
import mylib.labelme_utils as lu
import solaris as sol
import solaris.preproc.pipesegment as pipesegment
import solaris.preproc.label as label
import solaris.preproc.image as image
import solaris.preproc.sar as sar
from solaris.data import data_dir
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union


''' existing path '''
path = r'/home/csl/code/preprocess/data/SN6_extend'
slc_path = osp.join(path, r'newSLC2')
rgb_file_path = osp.join(path,
                    r'PS-RGB/SN6_AOI_11_Rotterdam_PS-RGB_TrainVal.tif')
label_files_path = [osp.join(path, r'geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_GT20sqm-TrainVal.geojson'), 
                    osp.join(path, r' geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_GT20sqm-Test.geojson')]
label_file_path = osp.join(path, r'geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_GT20sqm-TrainValTest.geojson')
tmp_dir = r'/home/csl/code/preprocess/tmp'

''' dirs of output files '''
tile_size = 900
tile_slc_path = osp.join(path, r'tile_slc', str(tile_size))
tile_rgb_path = osp.join(path, r'tile_rgb', str(tile_size))
tile_label_path = osp.join(path, r'tile_label', str(tile_size))
tile_mask_path = osp.join(path, r'tile_mask', str(tile_size))
tile_pauli_path = osp.join(path, r'tile_pauli', str(tile_size))
os.makedirs(tile_slc_path, exist_ok=True)
os.makedirs(tile_rgb_path, exist_ok=True)
os.makedirs(tile_label_path, exist_ok=True)
os.makedirs(tile_mask_path, exist_ok=True)
os.makedirs(tile_pauli_path, exist_ok=True)

''' tile PS-RGB '''
print(f'{LINE_SEPARATOR}\ntile PS-RGB\n{LINE_SEPARATOR}\n')
rgb_tiler = sol.tile.raster_tile.RasterTiler(
            dest_dir=tile_rgb_path,
            src_tile_size=(tile_size, tile_size),
            verbose=True)
rgb_bounds_crs = rgb_tiler.tile(rgb_file_path)

''' tile label in geojson format '''
print(f'{LINE_SEPARATOR}\ntile geojson label\n{LINE_SEPARATOR}\n')
vector_tiler = sol.tile.vector_tile.VectorTiler(
                dest_dir=tile_label_path,
                verbose=True)
vector_tiler.tile(label_file_path,
            tile_bounds=rgb_tiler.tile_bounds,
            tile_bounds_crs=rgb_bounds_crs)
            
''' change geojson to mask '''
print(f'{LINE_SEPARATOR}\nchange geojson label to mask\n{LINE_SEPARATOR}\n')
labels = glob(osp.join(tile_label_path, r'*.geojson'))
label_prefix = r'geoms_'
label_suffix = r'geojson'
rgb_prefix = r'SN6_AOI_11_Rotterdam_PS-RGB_TrainVal_'
rgb_suffix = r'tif'
PALETTE = np.array([[0, 0, 0], [255, 255, 255]])
for label in labels:
    rgb = label.replace(label_prefix, rgb_prefix) \
                .replace(label_suffix, rgb_suffix) \
                .replace(tile_label_path, tile_rgb_path)
    mask = sol.vector.mask.footprint_mask(df=label, reference_im=rgb)
    mask = mask > 0
    mask_path = osp.join(tile_mask_path, 
                    osp.basename(label).replace(label_suffix, 'png'))
    # print(f'saving {mask_path}')
    lu.lblsave(mask_path, mask, colormap=PALETTE)

''' tile SAR data in slc format with respect to PS-RGB '''
print(f'{LINE_SEPARATOR}\ntile SAR slc data\n{LINE_SEPARATOR}\n')
slcs = glob(osp.join(slc_path, r'slc_*.tif'))
print(f'{len(slcs)} samples in total')
slc_tiler = sol.tile.raster_tile.RasterTiler(
    dest_dir=tile_slc_path,
    verbose=True,
    # src_tile_size=(tile_size, tile_size),
    resampling='bilinear',
    tile_bounds=rgb_tiler.tile_bounds,
)
for slc in slcs:
    # timestamp = re.findall(r'\d{14}\_\d{14}', slc)
    # assert len(timestamp)==1, f'find {len(timestamp)} matches'
    # timestamp = timestamp[0]
    slc_tiler.tile(slc)