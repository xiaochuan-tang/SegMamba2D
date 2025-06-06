#!/bin/bash

# 输入文件夹路径
INPUT_DIR="datasets/HR-GLDD/test_data_png"
# 输出文件夹路径
OUTPUT_DIR="demo/hrgldd_predict"

# 如果输出文件夹不存在，则创建它
mkdir -p "$OUTPUT_DIR"

# 遍历输入文件夹中的所有图像文件（假设是 .png 格式）
for img_file in "$INPUT_DIR"/*.png; do
    # 获取图像文件的文件名，不包含路径
    filename=$(basename "$img_file")
    
    # 设置输出文件路径
    output_file="$OUTPUT_DIR/$filename"
    
    # 调用 Python 脚本进行预测
    python demo/image_demo.py \
        --img "$img_file" \
        --out "$output_file"    \
        --config tools/sh/hrgldd/hrgldd_B3.py  \
        --checkpoint output/hrgldd/segmamba_B3/iter_7200_miou_0.7184.pth
        # --config output/luding/segmambaB3_v2/luding.py  \
        # --checkpoint output/luding/segmambaB3_v2/iter_14400_miou0.693.pth
    
    # 打印处理的文件名
    echo "Processed $filename, saved to $output_file"
done

echo "所有图像预测完成，结果保存在 $OUTPUT_DIR 文件夹中。"
