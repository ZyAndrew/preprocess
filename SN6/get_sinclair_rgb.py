'''
Author: Shuailin Chen
Created Date: 2021-09-16
Last Modified: 2021-09-16
	content: get SinclairRGB, and compare with results of linear stretch (5% and 95% percentile), actually the differences between them are minor
'''

import os
import os.path as osp
import mylib.polSAR_utils as psr
import mylib.file_utils as fu
import mylib.image_utils as iu
import tifffile


def get_sinclair_rgb(src_dir, dst_dir, linear_version_dir):
    intens = os.listdir(src_dir)
    print(f'totally {len(intens)} samples')

    for inten in intens:
        ori = tifffile.imread(osp.join(src_dir, inten))
        sinclair = psr.rgb_by_s2(ori.transpose(2, 0 ,1), type='sinclair', if_mask=True)
        dst_path = osp.join(dst_dir, inten)
        linea_path = osp.join(linear_version_dir, inten)
        print(f'writting Sinclair RGB: {dst_path}\n linear version: {linea_path}')
        iu.save_image_by_cv2(sinclair, dst_path)


if __name__ == '__main__':
    src_dir = r'/home/csl/code/preprocess/data/SN6_full/SAR-Intensity'
    dst_dir = r'/home/csl/code/preprocess/data/SN6_full_sinclair'
    linear_version_dir = r'/home/csl/code/preprocess/data/SN6_full/SAR-PRO'
    get_sinclair_rgb(src_dir, dst_dir, linear_version_dir)