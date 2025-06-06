import cv2
import numpy as np
import os

# 文件夹路径
folder_path = r'datasets/GDCLD/annotations/validation'

# 输出文件夹路径
output_folder_path = r'datasets/GDCLD/annotations/validation_look'
os.makedirs(output_folder_path, exist_ok=True)

# 遍历文件夹中的所有图像文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)

        # 读取灰度图像
        gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # 将像素值为0的部分设为黑色，像素值为1的部分设为白色
        black_white_image = np.where(gray_image == 0, 0, 255)

        # 保存黑白图像
        output_file_path = os.path.join(output_folder_path, file)
        cv2.imwrite(output_file_path, black_white_image)

print("所有图像已转换为黑白图像并保存到指定文件夹。")