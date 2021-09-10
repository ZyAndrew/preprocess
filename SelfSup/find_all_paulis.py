'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-10
	content: find all pauliRGBs
'''


import os
import os.path as osp
import mylib.file_utils as fu
from glob import glob


def find_all_paulis(src_path):
    print(f'finding pauliRGBs in {src_path}')
    os.makedirs(osp.join(src_path, 'split'), exist_ok=True)

    assert osp.isdir(osp.join(src_path, 'data'))

    paulis = glob(osp.join(src_path, r'**/C3/*/PauliRGB.bmp'), recursive=True)
    fu.write_file_from_list(paulis, 
                            osp.join(src_path, 'split', 'all_paulis.txt'))
    print(f'done, {len(paulis)} samples in total')


if __name__ == '__main__':
    src_path = r'data/SAR_CD/RS2'
    src_path = r'data/SAR_CD/GF3'
    find_all_paulis(src_path)