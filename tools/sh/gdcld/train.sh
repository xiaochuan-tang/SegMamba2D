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
python tools/train.py \
    --config output/GDCLD/segmamba_B4_v2_ratio_range2.0/GDCLD_batchsize1.py \
    --work-dir output/GDCLD/segmamba_B4_v2_ratio_range2.0 \
    --load-from checkpoints/segformer.b4.1024x1024.city.160k.pth