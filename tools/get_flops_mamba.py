import argparse
import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string
from mmseg.models import build_segmentor
from fvcore.nn import FlopCountAnalysis


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        '--config', 
        default='output/GDCLD/segmambaB3_v2/GDCLD.py', 
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1024, 1024],
        help='Input image size (height, width)'
    )
    return parser.parse_args()


def compute_mamba_flops(h, w, block, seq_len):
    """计算单个 Mamba Block 的 FLOPs。"""
    flops = 0
    mamba_block = block.mamba_block

    # in_proj
    flops += 2 * seq_len * mamba_block.mixer.in_proj.in_features * mamba_block.mixer.in_proj.out_features

    # conv1d
    conv1d = mamba_block.mixer.conv1d
    flops += 2 * conv1d.out_channels * seq_len * conv1d.kernel_size[0] * (conv1d.in_channels / conv1d.groups)

    # x_proj
    flops += 2 * seq_len * mamba_block.mixer.x_proj.in_features * mamba_block.mixer.x_proj.out_features

    # dt_proj
    flops += 2 * seq_len * mamba_block.mixer.dt_proj.in_features * mamba_block.mixer.dt_proj.out_features

    # conv1d_b
    conv1d_b = mamba_block.mixer.conv1d_b
    flops += 2 * conv1d_b.out_channels * seq_len * conv1d_b.kernel_size[0] * (conv1d_b.in_channels / conv1d_b.groups)

    # x_proj_b
    flops += 2 * seq_len * mamba_block.mixer.x_proj_b.in_features * mamba_block.mixer.x_proj_b.out_features

    # dt_proj_b
    flops += 2 * seq_len * mamba_block.mixer.dt_proj_b.in_features * mamba_block.mixer.dt_proj_b.out_features

    # out_proj
    flops += 2 * seq_len * mamba_block.mixer.out_proj.in_features * mamba_block.mixer.out_proj.out_features

    # LayerNorm
    norm_features = mamba_block.norm.normalized_shape[0]
    flops += 5 * seq_len * norm_features

    return flops


def mamba_stage_flops(h, w, stage_blocks, num_blocks):
    """计算特定阶段中所有 Mamba Block 的 FLOPs。"""
    total_flops = 0
    seq_len = h * w
    for block in stage_blocks[:num_blocks]:
        total_flops += compute_mamba_flops(h, w, block, seq_len)
    return total_flops


def get_tr_flops_mamba(net, input_shape):
    """计算包含 Mamba Block 的模型 FLOPs。"""
    input = torch.randn(1, *input_shape).cuda()
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)

    _, H, W = input_shape
    net = net.backbone

    stage1_flops = mamba_stage_flops(H // 4, W // 4, net.block1, len(net.block1))
    stage2_flops = mamba_stage_flops(H // 8, W // 8, net.block2, len(net.block2))
    stage3_flops = mamba_stage_flops(H // 16, W // 16, net.block3, len(net.block3))
    stage4_flops = mamba_stage_flops(H // 32, W // 32, net.block4, len(net.block4))

    total_mamba_flops = stage1_flops + stage2_flops + stage3_flops + stage4_flops
    total_flops = flops + total_mamba_flops

    print(f"Mamba Block FLOPs: {total_mamba_flops}")
    print(f"Total FLOPs: {total_flops}")

    return flops_to_string(total_flops), params_to_string(params)


def main():
    """主函数，计算模型 FLOPs 和参数量。"""
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('Invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    ).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            f'FLOPs counter is not supported for {model.__class__.__name__}'
        )

    if hasattr(model.backbone, 'block1'):
        print('#### Calculate FLOPs with Mamba Block ####')
        flops, params = get_tr_flops_mamba(model, input_shape)
    else:
        print('#### Calculate FLOPs for CNN ####')
        flops, params = get_model_complexity_info(model, input_shape)

    print('=' * 30)
    print(f"Input shape: {input_shape}")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    print('=' * 30)
    print('!!! Verify FLOPs calculation when using in papers.')


if __name__ == '__main__':
    main()
