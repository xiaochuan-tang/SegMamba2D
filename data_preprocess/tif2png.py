from PIL import Image
import os

tif_dir = 'datasets/GDCLD/val_dataset/val_label_tif'
png_dir = 'datasets/GDCLD/val_dataset/val_label_png'

os.makedirs(png_dir, exist_ok=True)

for tif_file in os.listdir(tif_dir):
    if tif_file.endswith('.tif'):
        img = Image.open(os.path.join(tif_dir, tif_file))
        # 去掉文件名中的 '_t' 并将后缀改为 .png
        png_file = tif_file.replace('_s', '').replace('.tif', '.png')
        img.save(os.path.join(png_dir, png_file))
