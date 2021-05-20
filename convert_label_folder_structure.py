'''
Author: Shuailin Chen
Created Date: 2021-01-22
Last Modified: 2021-05-20
	content: 
'''
''' 用于极化SAR的数据集制备，将赵嘉霖标注的文件夹的结构换成我的，要重复用的话可能要改一下代码 
'''

import shutil 
import os.path as osp
import os
import re
import time
from mylib import labelme_utils as lu

def mkdir_if_not_exist(path):  
    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)


def extract_time_and_convert(name:str)->str:
    ''' extract time from a string and convert to a standard form, like yyyy/mm/dd'''
    tm_ori = re.search('\d{2}[a-zA-Z]{3}\d{4}', name)
    if tm_ori is not None:
        tm = tm_ori.group()
        tm = time.strptime(tm, '%d%b%Y')
        tm = time.strftime('%Y%m%d', tm)
    else:
        tm = re.search('\d{8}', name)
        tm = tm.group()
        if tm is None:
            raise NotImplementedError
    return tm


if __name__ == '__main__':
    path = r'/home/csl/code/preprocess/data/SAR_CD/GF3/label/E115_N39_中国河北/升轨/1'

    dates = ['20190627', '20170308']
    mkdir_if_not_exist(osp.join(path, dates[0]))
    mkdir_if_not_exist(osp.join(path, dates[1]))
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-5:] == '.json':
                tm = [item for item in dates if file[:4] in item][0]       #提取时间
                num = file[-8:-5]
                old_path = osp.join(root, file)
                new_full_path = osp.join(path, tm, tm+'_'+num+'.json')
                if old_path != new_full_path:
                    print('copy', old_path, 'to', new_full_path)
                    shutil.copy(old_path, new_full_path)
                    shutil.copy(old_path, new_full_path.replace('json','png'))
                # old_path = old_path[:-4] + 'png'  
