'''
Author: Shuailin Chen
Created Date: 2021-01-23
Last Modified: 2021-04-25
'''
''' 重命名 极化SAR数据的目录，方便后续处理
    1) 目录名字仅保留时间信息，其余的产品信息打包成一个 .txt 文件 
'''
import os
import re
import os.path as osp
import time
import shutil
from mylib import polSAR_utils

def rm_part_of_foldername(path:str, part:str):
    ''' 将文件夹中的某个部分去掉去掉 '''
    for root, dirs, files in os.walk(path):
        for file in files:
            if part in file:
                new_name = file.replace(part, '')
                os.rename(osp.join(root, file), osp.join(root, new_name))
                print('rename', file, 'to', new_name)


def rename_time_format_of_file_name(path:str):
    ''' 修改文件名中的时间格式 '''
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-5:] == '.json':
                tm_ori, tm_cvt = extract_time_and_convert(file)
                old_path = osp.join(root, file)
                new_path = osp.join(root, file.replace(tm_ori, tm_cvt))
                print('rename ', file, ' to ', file.replace(tm_ori, tm_cvt))
                os.rename(old_path, new_path)

    print('done')


def rename_time_format_of_folder_name(path:str):
    ''' 修改文件夹名字中的时间格式 '''
    for folder in os.listdir(path):
        if osp.isdir(osp.join(path, folder)):
            tm_ori, tm_cvt = extract_time_and_convert(folder)
            old_path = osp.join(path, folder)
            new_path = osp.join(path, folder.replace(tm_ori, tm_cvt))
            print('rename ', folder, ' to ', folder.replace(tm_ori, tm_cvt))
            os.rename(old_path, new_path)
    
    print('done')


def extract_time_and_convert(name:str)->str:
    ''' extract time from a string and convert to a standard form, like yyyy/mm/dd'''
    tm_ori = re.search('\d{2}[a-zA-Z]{3}\d{4}', name)
    if tm_ori is not None:
        tm_ori = tm_ori.group()
        tm_cvt = time.strptime(tm_ori, '%d%b%Y')
        tm_cvt = time.strftime('%Y%m%d', tm_cvt)
    else:
        tm_cvt = re.search('\d{8}', name)
        tm_cvt = tm_cvt.group()
        if tm_cvt is None:
            raise NotImplementedError
        tm_ori = tm_cvt
    return tm_ori, tm_cvt


if __name__=='__main__':
    path = r'/home/csl/code/preprocess/data/SAR_CD/GF3/data/E139_N35_日本横滨'
    for root, dirs, files in os.walk(path):
        if 'C3' in dirs:
            details = root.split('/')[-1]
            tm = re.search('\d{8}', details).group()
            with open(osp.join(root, 'README.txt'), 'w') as f:
                f.write(f'详细的产品信息为: {details}, 文件夹名字仅包含了时间信息')
            print(f'rename {details} to {tm}', end='')
            os.rename(root, root.replace(details, tm))
            print('\tdone')

    print('all done')