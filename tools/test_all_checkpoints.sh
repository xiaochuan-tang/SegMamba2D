#!/bin/bash

# 配置文件路径
# local_configs/segformer/B3/luding.py
# local_configs/segformer/B4/GDCLD.py
# CONFIG="local_configs/segformer/B3/seg_hr_mamba.py"
# CONFIG="output/luding/segmambaB3_v2/luding.py"
# CONFIG="output/GDCLD/segmambaB4_v2/GDCLD.py"
CONFIG="output/GDCLD/segmamba_B4_v2_ratio_range2.0/GDCLD_batchsize1.py"




# Checkpoint 文件夹路径
# output/luding/segformerB3/seg_hr_mamba
# output/GDCLD/segmamba_b4
# output/luding/segmambav4
# CHECKPOINT_DIR="output/luding/segmambaB3_v2"
CHECKPOINT_DIR="output/GDCLD/segmamba_B4_v2_ratio_range2.0"


# 输出结果文件路径
OUTPUT_DIR="work_dirs"

# 日志文件路径
# log/seg_hr_mamba
# log/segmamba_B4/output.log
LOG_FILE="log/gdcld/segmamba_B4_v2_ratio_range2.log"

# 将标准输出和标准错误输出都重定向到日志文件
exec > >(tee -a ${LOG_FILE}) 2>&1

# 遍历所有 .pth 文件
for CHECKPOINT in ${CHECKPOINT_DIR}/*.pth; do
    # 提取 checkpoint 文件名
    CHECKPOINT_NAME=$(basename ${CHECKPOINT})
    # 生成输出文件路径
    OUTPUT_FILE="${OUTPUT_DIR}/${CHECKPOINT_NAME%.pth}_res.pkl"
    
    # 执行 Python 脚本
    python tools/test.py --config ${CONFIG} \
                   --checkpoint ${CHECKPOINT} \
                   --out ${OUTPUT_FILE} \
                   --eval mIoU

    echo "Finished testing ${CHECKPOINT_NAME}, results saved to ${OUTPUT_FILE}"

    # 等待 2 秒
    sleep 2
done
