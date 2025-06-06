# import pickle

# # 打开 pkl 文件
# with open('work_dirs/res.pkl', 'rb') as file:
#     data = pickle.load(file)

# # 查看数据类型和内容
# print(type(data))
# print(data)


import pickle
import numpy as np

# 指定要读取的 .pkl 文件路径
pkl_file = 'work_dirs/res.pkl'

# 打开并读取 .pkl 文件
with open(pkl_file, 'rb') as file:
    npy_files = pickle.load(file)

# 遍历并加载 .npy 文件
for npy_file in npy_files:
    data = np.load(npy_file)
    print(f"Data from {npy_file}:")
    print(data)
    print("\n")
