'''
Author: Shuailin Chen
Created Date: 2021-09-04
Last Modified: 2021-09-04
	content: further pro
'''

import os
import os.path as osp
from glob import glob
import mylib.file_utils as fu
import shutil


def rm_ext(split_folder):
    ''' Remove external name of each item in split files '''

    for file in glob(osp.join(split_folder, '*/*.txt')):
        print(f'process {file}')
        items = fu.read_file_as_list(file)
        items = [osp.splitext(ii)[0] for ii in items]
        fu.write_file_from_list(items, file)


def check_npy_split(split_folder, npy_folder):
    ''' Check the real split in npy folder wether corresponding the split file, if not, move it according to the split file
    '''
    
    # read split files
    splits = dict()
    for sensor in os.listdir(split_folder):
        for split in os.listdir(osp.join(split_folder, sensor)):
            if splits.get(osp.splitext(split)[0], None):
                splits[osp.splitext(split)[0]] += fu.read_file_as_list(
                                        osp.join(split_folder, sensor, split))
            else:
                splits[osp.splitext(split)[0]] = fu.read_file_as_list(
                                        osp.join(split_folder, sensor, split))

    # check the real file splits in the data folder
    for split in os.listdir(npy_folder):
        if osp.isfile(osp.join(npy_folder, split)):
            continue
        for file in os.listdir(osp.join(npy_folder, split)):
            if osp.splitext(file)[0] not in splits[split]:
                dst_split = None
                for split_, list_ in splits.items():
                    if osp.splitext(file)[0] in splits[split_]:
                        dst_nsplit =  split_
                        break

                if dst_split is None:
                    raise ValueError(f'can not find target split of file {file} with original split {split}')

                print(f'move {file} from {split} to {dst_split}')
                shutil.move(osp.join(npy_folder, split, file),
                            osp.join(npy_folder, dst_split, file))





if __name__ == '__main__':
    split_folder = r'data/ade20k/sar_building/split'
    npy_folder = r'data/ade20k/sar_building/npy'
    # rm_ext(split_folder)
    check_npy_split(split_folder, npy_folder)
