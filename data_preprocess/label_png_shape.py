from PIL import Image
import numpy as np

# 打开mask文件
mask = Image.open('datasets/GDCLD/annotations/training/00005.png')

# 将mask转换为numpy数组
mask_array = np.array(mask)

# 查看mask的尺寸
print("Mask shape:", mask_array.shape)

# 查看像素值的分布
unique_values, counts = np.unique(mask_array, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"Pixel value {value}: {count} pixels")
