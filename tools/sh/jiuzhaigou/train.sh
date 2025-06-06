#!/bin/bash


# python tools/train.py \
#     --config local_configs/segformer/B3/GDCLD.py \
#     --work-dir output/GDCLD/segmambaB3_v2_200k  \
#     --resume-from output/GDCLD/segmambaB3_v2/iter_140000.pth

#     --config local_configs/segformer/B4/GDCLD.py \
#     --config output/GDCLD/segmamba_b4/GDCLD.py

# python tools/train.py \
#     --config local_configs/segformer/B4/GDCLD.py \
#     --work-dir output/GDCLD/segmambaB4_v2 \
#     --load-from checkpoints/segformer.b4.1024x1024.city.160k.pth


# segformer_b3
# output/GDCLD/segmambaB3_v2/GDCLD.py
# python tools/train.py \
#     --config tools/sh/jiuzhaigou/jiuzhaigou_B3.py \
#     --work-dir output/jiuzhaigou/segformer_B3_repeat1
#     # --resume-from output/jiuzhaigou/segmambaB3_v2/iter_13400.pth
#     # --load-from checkpoints/segformer.b4.1024x1024.city.160k.pth


# python tools/train.py \
#     --config tools/sh/jiuzhaigou/jiuzhaigou_B3_resize512.py \
#     --work-dir output/jiuzhaigou/segmambaB3_v2_resize512

python tools/train.py \
    --config tools/sh/jiuzhaigou/jiuzhaigou_B3.py \
    --work-dir output/jiuzhaigou/test