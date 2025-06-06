from argparse import ArgumentParser
from PIL import Image
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    # datasets/mamba_opt/images/validation/94.png
    # datasets/GDCLD/images/validation/00009.png
    
    parser.add_argument('--img', default='datasets/mamba_opt/images/validation/94.png', help='Image file')
    # local_configs/segformer/B3/segformer.b3.1024x1024.city.160k.py
    parser.add_argument('--config', default='output/luding/segmambaB3_v2/luding.py', help='Config file')
    parser.add_argument('--checkpoint', default='output/luding/segmambaB3_v2/iter_14400_miou0.693.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument('--out', default='demo/luding_94.png', help='Output file to save the result')

    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    print(result[0].shape)
    # # 使用 np.unique 获取唯一值
    # unique_values = np.unique(result[0])
    # print("Array 中的唯一值:", unique_values)

    # # 如果你还想查看每个值的出现次数，可以这样使用：
    # unique_values, counts = np.unique(result[0], return_counts=True)
    # print("唯一值及其出现次数:", dict(zip(unique_values, counts)))

    # 转换为8位整型
    result_img = Image.fromarray(result[0].astype(np.uint8))
    result_img.save(args.out)
    print(f"Result saved to {args.out}")

    # show the results
    # # show_result_pyplot(model, args.img, result, get_palette(args.palette))
    # show_result_pyplot(model, args.img, result, get_palette(args.palette), out_file=args.out)



if __name__ == '__main__':
    main()
