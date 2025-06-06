# from PIL import Image

# # 打开已保存的灰度图像
# gray_image = Image.open("demo/luding_94.png").convert("L")  # 确保是灰度图像

# # 设置阈值并转换为黑白图像
# threshold = 0  # 可调整阈值
# black_white_image = gray_image.point(lambda x: 255 if x > threshold else 0, mode="1")

# # 保存为黑白图像
# black_white_image.save("demo/luding_94_bw.png")
# print("黑白图像已保存！")

import os
from PIL import Image

def convert_folder_to_black_white(input_folder, output_folder, threshold=0):
    """
    将文件夹中的所有灰度图像转换为黑白图像。

    Args:
        input_folder (str): 输入灰度图像文件夹路径。
        output_folder (str): 输出黑白图像文件夹路径。
        threshold (int): 阈值，像素值高于该值设为白色，低于该值设为黑色。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 创建输出文件夹

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # 打开灰度图像
                gray_image = Image.open(input_path).convert("L")  # 确保是灰度图像

                # 设置阈值并转换为黑白图像
                black_white_image = gray_image.point(lambda x: 255 if x > threshold else 0, mode="1")

                # 保存黑白图像到输出文件夹
                output_path = os.path.join(output_folder, filename)
                black_white_image.save(output_path)
                print(f"转换完成: {filename} -> {output_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

# 设置输入和输出文件夹路径
input_folder = "datasets/HR-GLDD/test_label_png"  # 替换为你的输入文件夹路径
output_folder = "demo/test_label_png_look"  # 替换为你的输出文件夹路径

# 调用函数，将灰度图像转换为黑白图像
convert_folder_to_black_white(input_folder, output_folder, threshold=0)
