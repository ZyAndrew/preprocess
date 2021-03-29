'''
Author: Shuailin Chen
Created Date: 2021-01-22
Last Modified: 2021-03-24
	content: 
'''
''' 重命名 labelme 标注的文件，方便后续的处理 
last modefied: 2020-01-23
'''

import os
import re
import os.path as osp
import time
import shutil
from mylib import labelme_utils as lu

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


def regular_folder_structure(path:str):
    ''' 将文件夹格式转换成规定的格式，
        即： 1) 将文件夹的名字改成日期
             2) 将文件名改为文件夹名+序号
    '''
    print('processing folder:', path)
    for folder in os.listdir(path):
        if osp.isdir(osp.join(path, folder)):
            tm_ori, tm_cvt = extract_time_and_convert(folder)
            old_path = osp.join(path, folder)
            new_path = osp.join(path, tm_cvt)
            print('rename ', folder, ' to ', tm_cvt)
            os.rename(old_path, new_path)

            for file in os.listdir(new_path):
                filename, file_ext = osp.splitext(file)
                if file_ext in ('.png', '.json'):
                    # 提取序号
                    num = re.search('\d{3}', filename).group()
                    new_name = tm_cvt + '_' + num + file_ext
                    old_path = osp.join(path, tm_cvt, file)
                    new_path = osp.join(path, tm_cvt, new_name)
                    print('rename ', file, ' to ', new_name)
                    os.rename(old_path, new_path)

    print('done')





if __name__=='__main__':
    # path = r'/data/csl/SAR_CD/RS2/data/湛江/'
    # rm_part_of_foldername(path, r'PauliRGB-')

    
    path = r'data/SAR_CD/GF3/label/E130_N34_日本鞍手/降轨/4'
    regular_folder_structure(path)
    for fdr in os.listdir(path):
        fdr = osp.join(path, fdr)
        if osp.isdir(fdr):
            regular_folder_structure(fdr)

    # path = r'/data/csl/SAR_CD/GF3/label/E115_N39_中国河北/降轨/1/20170306/'
    # for file in os.listdir(path):
    #     filename, file_ext = osp.splitext(file)
    #     if file_ext in ('.png', '.json'):
    #         # 提取序号
    #         num = re.search('\d{3}', filename).group()
    #         new_name = r'20170306' + '_' + num + file_ext
    #         old_path = osp.join(path, file)
    #         new_path = osp.join(path, new_name)
    #         print('rename ', file, ' to ', new_name)
    #         os.rename(old_path, new_path)

    # # label文件中的时间标错了，这里将标错的20190625改成20190615
    # path = r'/data/csl/SAR_CD/GF3/label/E139_N35_日本横滨/降轨/1/'
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if '20190625' in file:
    #             old_name = osp.join(root, file)
    #             new_name = old_name.replace('20190625', '20190615')
    #             print(f'rename {old_name} to {new_name}')
    #             os.rename(old_name, new_name)

    print('done')