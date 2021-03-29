import os
import os.path as osp

path = r'/home/csl/code/preprocess/data/SAR_CD/GF3/'

cnt = 0
with open(osp.join(path, 'train.txt'), 'w') as f:
    for root, dirs, files in os.walk(path):
        for file in files:
            if '-change.png' in file:
                rel_path = osp.join(root, file).replace(path, '')
                f.write(rel_path+'\n')
                cnt += 1

print(f'totally {cnt} files')