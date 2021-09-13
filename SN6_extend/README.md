
# 处理spacenet6 extend的数据集

原始数据集包含SLC数据，并且没有对图像进行裁剪

完整的处理原始数据的pipeline为：
1. extract_slc_to_one_file.py：将四个通道的数据集合到一张图像中
2. tile.py：将原始的大图裁剪成小图，参考图像为PS-RGB
3. post_process.py：后处理，生成SAR图像的pauliRGB，去除无效的数据和标注