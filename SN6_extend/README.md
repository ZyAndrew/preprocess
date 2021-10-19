
# 处理spacenet6 extend的数据集

原始数据集包含SLC数据，并且没有对图像进行裁剪

完整的处理原始数据的pipeline为：
1. `extract_slc_to_one_file.py`：将四个通道的SAR原始数据集合到一张图像中
2. `tile.py`：将原始的大图裁剪成小图，参考图像为PS-RGB
3. `post_process.py`：后处理，PS-RGB在裁剪的时候会出现有效区域仅有一行或者两行的情况，需要去除；生成SAR图像的pauliRGB，同样去除无效的数据和标注，
4. `post_process2.py`：有效的mask和有效的PS-RGB的patch并不完全重叠，需要再计算其交集
5. `merge_duplicate_label.py`: mask文件夹中同一个地理patch可能同时存在于train和val，并且各自只有一部分的标注，因此将其合起来，并全部放在训练集中
6. `get_sinclair_v2`：生成类似郭浩文预处理的图像，也即将上限和下限设为[5%, 95%]，然后现行归一化。选取的三个波段是HH，VV，VH

#
