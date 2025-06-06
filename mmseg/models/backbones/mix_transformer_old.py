# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math

# import mmseg.models.backbones.models_mamba as models_mamba
import mmseg.Vim.vim.models_mamba as models_mamba

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.mamba_block = None
        # bimamba_type = "v1"
        # if_devide_out=False
        # init_layer_scale = None
        # ssm_cfg ={}
        # factory_kwargs={}
        # mixer_cls = partial(Mamba,layer_idx=None,bimamba_type=bimamba_type,init_layer_scale=init_layer_scale,**ssm_cfg,**factory_kwargs)

        self.mamba_block = models_mamba.create_block(d_model=dim).to("cuda") # 原segmamba
        # self.mamba_block = models_mamba.create_block(d_model=dim, bimamba_type = 'v2').to("cuda")


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # mit_b0 mit_b1 mit_b2 mit_b3 mit_b4 唯一的区别在于depths，表示在每个阶段中有多少个 Transformer 块。
        # 预计对mamba不会有负面影响
        
        # print("h")
        # import pdb; pdb.set_trace()

        x = self.norm1(x)
        # norm = self.norm1(x)

        # _, _, C = norm.shape
        # print(C)
        # if self.mamba_block is None:
            # self.mamba_block = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block(x)
        x = x + self.drop_path(hidden_states)

        # x = x + self.drop_path(self.attn(self.norm1(x), H, W))

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class FuseLayers(nn.Module):
    def __init__(self, embed_dims):
        super(FuseLayers, self).__init__()
        self.num_branches = len(embed_dims)
        self.fuse_layers = self._make_fuse_layers(embed_dims)

    def _make_fuse_layers(self, embed_dims):
        num_branches = len(embed_dims)
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # 上采样部分保持不变
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(embed_dims[j], embed_dims[i], kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(embed_dims[i], momentum=0.1),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # 将深度可分离卷积修改为空洞卷积
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = embed_dims[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    # 深度可分离卷积
                                    # nn.Conv2d(embed_dims[j], embed_dims[j], kernel_size=3, stride=2, padding=1, groups=embed_dims[j], bias=False),
                                    # nn.BatchNorm2d(embed_dims[j], momentum=0.1),
                                    # nn.Conv2d(embed_dims[j], num_outchannels_conv3x3, kernel_size=1, stride=1, bias=False),
                                    # nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.1),

                                    # 修改为空洞卷积，设置 dilation 参数
                                    nn.Conv2d(embed_dims[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.1),

                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    # 深度可分离卷积
                                    # nn.Conv2d(embed_dims[j], embed_dims[j], kernel_size=3, stride=2, padding=1, groups=embed_dims[j], bias=False),
                                    # nn.BatchNorm2d(embed_dims[j], momentum=0.1),
                                    # nn.Conv2d(embed_dims[j], embed_dims[j], kernel_size=1, stride=1, bias=False),
                                    # nn.BatchNorm2d(embed_dims[j], momentum=0.1),
                                    # nn.ReLU(False),
                                    nn.Conv2d(embed_dims[j], embed_dims[j], kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
                                    nn.BatchNorm2d(embed_dims[j], momentum=0.1),
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        out = []
        for i in range(len(x)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, len(x)):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            out.append(F.relu(y))
        return out


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.block1_1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[1])])
        self.norm1_1 = norm_layer(embed_dims[0])

        self.block1_2 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[2])])
        self.norm1_2 = norm_layer(embed_dims[0])

        self.block1_3 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[3])])
        self.norm1_3 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.block2_1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[2])])
        self.norm2_1 = norm_layer(embed_dims[1])

        self.block2_2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[3])])
        self.norm2_2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.block3_1 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[3])])
        self.norm3_1 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # mamba
        self.mamba_block1 = None
        self.mamba_block2 = None
        self.mamba_block3 = None
        self.mamba_block4 = None

        # Fuse layers
        # self.fuse_layers1 = FuseLayers([64, 128, 320])  #  
        # self.fuse_layers2 = FuseLayers([64, 128, 320, 512])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # print("x.shape:",x.shape)   # torch.Size([2, 3, 395, 395])
        # import pdb;pdb.set_trace()
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage1, x.shape:",x.shape)   # torch.Size([2, 64, 99, 99])

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage2, x.shape:",x.shape)   # torch.Size([2, 128, 50, 50])

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage3, x.shape:",x.shape)   # torch.Size([2, 320, 25, 25])

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage4, x.shape:",x.shape)   # torch.Size([2, 512, 13, 13])

        # import pdb;pdb.set_trace()
        return outs

    def forward_mamba(self, x):
        # print("x.shape:",x.shape)   # torch.Size([2, 3, 395, 395])
        # import pdb;pdb.set_trace()
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        _, _, C = x.shape
        if self.mamba_block1 is None:
            self.mamba_block1 = models_mamba.create_block(d_model=C).to("cuda")
        # print("x.shape:",x.shape)   # x.shape: torch.Size([2, 9801, 64])
        # print("H.shape:",H) # 99
        # print("W.shape:",W) # 99
        # import pdb;pdb.set_trace()
        
        # self.mamba_block1()
        # mamba = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block1(x)
        # print("mamba hidden_states.shape:", hidden_states.shape)    # torch.Size([2, 9801, 64])
        # for i, blk in enumerate(self.block1):
        #     x = blk(x, H, W)
        x = self.norm1(hidden_states)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage1, x.shape:",x.shape)   # torch.Size([2, 64, 99, 99])

        # stage 2
        x, H, W = self.patch_embed2(x)
        _, _, C = x.shape
        if self.mamba_block2 is None:
            self.mamba_block2 = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block2(x)
        # for i, blk in enumerate(self.block2):
        #     x = blk(x, H, W)
        x = self.norm2(hidden_states)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage2, x.shape:",x.shape)   # torch.Size([2, 128, 50, 50])

        # stage 3
        x, H, W = self.patch_embed3(x)
        _, _, C = x.shape
        if self.mamba_block3 is None:
            self.mamba_block3 = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block3(x)

        # for i, blk in enumerate(self.block3):
        #     x = blk(x, H, W)
        x = self.norm3(hidden_states)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage3, x.shape:",x.shape)   # torch.Size([2, 320, 25, 25])

        # stage 4
        x, H, W = self.patch_embed4(x)
        _, _, C = x.shape
        if self.mamba_block4 is None:
            self.mamba_block4 = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block4(x)

        # for i, blk in enumerate(self.block4):
        #     x = blk(x, H, W)
        x = self.norm4(hidden_states)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage4, x.shape:",x.shape)   # torch.Size([2, 512, 13, 13])

        # import pdb;pdb.set_trace()
        return outs

    def forward_mamba_v4(self, x):
        # print("x.shape:",x.shape)   # torch.Size([2, 3, 395, 395])
        # import pdb;pdb.set_trace()
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        _, _, C = x.shape
        if self.mamba_block1 is None:
            self.mamba_block1 = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block1(x)
        x = hidden_states.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage1, x.shape:",x.shape)   # torch.Size([2, 64, 99, 99])

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        _, _, C = x.shape
        if self.mamba_block2 is None:
            self.mamba_block2 = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block2(x)
        x = hidden_states.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage2, x.shape:",x.shape)   # torch.Size([2, 128, 50, 50])

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        _, _, C = x.shape
        if self.mamba_block3 is None:
            self.mamba_block3 = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block3(x)
        x = hidden_states.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage3, x.shape:",x.shape)   # torch.Size([2, 320, 25, 25])

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        _, _, C = x.shape
        if self.mamba_block4 is None:
            self.mamba_block4 = models_mamba.create_block(d_model=C).to("cuda")
        hidden_states, residual = self.mamba_block4(x)
        x = hidden_states.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        # print("stage4, x.shape:",x.shape)   # torch.Size([2, 512, 13, 13])

        # import pdb;pdb.set_trace()
        return outs

    
    def forward_seg_hr_mamba(self, x):
        # print("x.shape:",x.shape)   # torch.Size([2, 3, 512, 512])
        # x = F.interpolate(x, size=(472, 472), mode='bilinear', align_corners=False)
        # import pdb;pdb.set_trace()
        B = x.shape[0]
        outs = []

        # stage 1
        x_1, H_1, W_1 = self.patch_embed1(x)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        for i, blk in enumerate(self.block1):
            x_1 = blk(x_1, H_1, W_1)
        x_1 = self.norm1(x_1)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        x_1_reshape = x_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 64, 118, 118])
        # outs.append(x_1_reshape)
        # print("stage1, x.shape:",x.shape)   #torch.Size([2, 64, 118, 118])

        # mamba block 1_1
        for i, blk in enumerate(self.block1_1):
            x_1_1 = blk(x_1, H_1, W_1)
        x_1_1 = self.norm1_1(x_1_1)
        # print("x_1_1.shape:",x_1_1.shape)   # torch.Size([2, 16384, 64])
        x_1_1_reshape = x_1_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_1_1_reshape.shape:",x_1_1_reshape.shape)   # torch.Size([2, 64, 128, 128])

        # stage 2
        x_2, H_2, W_2 = self.patch_embed2(x_1_reshape)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        for i, blk in enumerate(self.block2):
            x_2 = blk(x_2, H_2, W_2)
        x_2 = self.norm2(x_2)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        x_2_reshape = x_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 128, 59, 59])
        # outs.append(x_2_reshape)
        # print("x_2_reshape.shape:",x_2_reshape.shape)   # torch.Size([2, 128, 64, 64])

        # Fuse x_1_1_reshape and x_2_reshape
        fused_outputs = self.fuse_layers1([x_1_1_reshape, x_2_reshape])

        # mamba block 1-2
        x_1_2 = fused_outputs[0]    # torch.Size([2, 64, 128, 128])
        x_1_2 = x_1_2.view(x_1_2.size(0), -1, x_1_2.size(1))    # torch.Size([2, 13924, 64])
        for i, blk in enumerate(self.block1_2):
            x_1_2 = blk(x_1_2, H_1, W_1)
        x_1_2 = self.norm1_2(x_1_2)
        x_1_2_reshape = x_1_2.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 64, 118, 118])
        # print("x_1_2_reshape.shape:",x_1_2_reshape.shape)   # torch.Size([2, 64, 128, 128])


        x_2_1 = fused_outputs[1]    # ([2, 128, 64, 64])
        x_2_1 = x_2_1.view(x_2_1.size(0), -1, x_2_1.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_1):
            x_2_1 = blk(x_2_1, H_2, W_2)
        x_2_1 = self.norm2_1(x_2_1)
        x_2_1_reshape = x_2_1.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        # print("x_2_1_reshape.shape:",x_2_1_reshape.shape)   # torch.Size([2, 128, 64, 64])




        # stage 3
        x_3, H_3, W_3 = self.patch_embed3(x_2_reshape)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        for i, blk in enumerate(self.block3):
            x_3 = blk(x_3, H_3, W_3)
        x_3 = self.norm3(x_3)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        x_3_reshape = x_3.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 320, 32, 32]) 
        # outs.append(x_3_reshape)
        # print("stage3, x_3.shape:",x_3_reshape.shape)   # torch.Size([2, 320, 32, 32]) 


        # Fuse x_1_2_reshape and x_2_1_reshape and x_3_reshape
        # torch.Size([2, 64, 128, 128])/torch.Size([2, 128, 64, 64])/torch.Size([2, 320, 32, 32]) 
        fused_outputs = self.fuse_layers2([x_1_2_reshape, x_2_1_reshape, x_3_reshape])

        x_1_3 = fused_outputs[0]
        # print("x_1_3.shape:",x_1_3.shape)  # torch.Size([2, 64, 128, 128])
        x_1_3 = x_1_3.view(x_1_3.size(0), -1, x_1_3.size(1))    
        for i, blk in enumerate(self.block1_3):
            x_1_3 = blk(x_1_3, H_1, W_1)
        x_1_3 = self.norm1_3(x_1_3)
        x_1_3_reshape = x_1_3.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_1_3_reshape.shape:",x_1_3_reshape.shape)   # torch.Size([2, 64, 128, 128])
        outs.append(x_1_3_reshape)


        x_2_2 = fused_outputs[1]
        # print(x_2_2.shape)  # torch.Size([2, 128, 64, 64])
        x_2_2 = x_2_2.view(x_2_2.size(0), -1, x_2_2.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_2):
            x_2_2 = blk(x_2_2, H_2, W_2)
        x_2_2 = self.norm2_2(x_2_2)
        x_2_2_reshape = x_2_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        # print("x_2_2_reshape.shape:",x_2_2_reshape.shape)   # torch.Size([2, 128, 64, 64])
        outs.append(x_2_2_reshape)

        x_3_1 = fused_outputs[2]
        # print(x_3_1.shape)  # torch.Size([2, 320, 32, 32])
        x_3_1 = x_3_1.view(x_3_1.size(0), -1, x_3_1.size(1))    # torch.Size([2, 32*32, 320])
        for i, blk in enumerate(self.block3_1):
            x_3_1 = blk(x_3_1, H_3, W_3)
        x_3_1 = self.norm3_1(x_3_1)
        x_3_1_reshape = x_3_1.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 320, 32, 32])
        # print("x_3_1_reshape.shape:",x_3_1_reshape.shape)   # torch.Size([2, 320, 32, 32])
        outs.append(x_3_1_reshape)







        # stage 4
        x_4, H_4, W_4 = self.patch_embed4(x_3_reshape)
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])
        for i, blk in enumerate(self.block4):
            x_4 = blk(x_4, H_4, W_4)
        x_4 = self.norm4(x_4)
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])
        x_4 = x_4.reshape(B, H_4, W_4, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 512, 15, 15])
        outs.append(x_4)
        # print("stage4, x_4.shape:",x_4.shape)   # torch.Size([2, 512, 15, 15])

        # import pdb;pdb.set_trace()
        return outs

    
    def forward_seg_hr_mamba_v2(self, x):
        # print("x.shape:",x.shape)   # torch.Size([2, 3, 512, 512])
        # x = F.interpolate(x, size=(472, 472), mode='bilinear', align_corners=False)
        # import pdb;pdb.set_trace()
        B = x.shape[0]
        outs = []

        # stage 1
        x_1, H_1, W_1 = self.patch_embed1(x)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        for i, blk in enumerate(self.block1):
            x_1 = blk(x_1, H_1, W_1)
        x_1 = self.norm1(x_1)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        x_1_reshape = x_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 64, 118, 118])
        # outs.append(x_1_reshape)
        # print("stage1, x.shape:",x.shape)   #torch.Size([2, 64, 118, 118])

        # mamba block 1_1
        for i, blk in enumerate(self.block1_1):
            x_1_1 = blk(x_1, H_1, W_1)
        x_1_1 = self.norm1_1(x_1_1)
        # print("x_1_1.shape:",x_1_1.shape)   # torch.Size([2, 16384, 64])
        x_1_1_reshape = x_1_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_1_1_reshape.shape:",x_1_1_reshape.shape)   # torch.Size([2, 64, 128, 128])

        # stage 2
        x_2, H_2, W_2 = self.patch_embed2(x_1_reshape)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        for i, blk in enumerate(self.block2):
            x_2 = blk(x_2, H_2, W_2)
        x_2 = self.norm2(x_2)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        x_2_reshape = x_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 128, 59, 59])
        # outs.append(x_2_reshape)
        # print("x_2_reshape.shape:",x_2_reshape.shape)   # torch.Size([2, 128, 64, 64])

        # stage 3
        x_3, H_3, W_3 = self.patch_embed3(x_2_reshape)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        x_3 = x_3.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 320, 32, 32]) 

        # Fuse x_1_1_reshape and x_2_reshape、x_3
        fused_outputs = self.fuse_layers1([x_1_1_reshape, x_2_reshape, x_3])


        # mamba block 1-2
        x_1_2 = fused_outputs[0]    # torch.Size([2, 64, 128, 128])
        x_1_2 = x_1_2.view(x_1_2.size(0), -1, x_1_2.size(1))    # torch.Size([2, 13924, 64])
        for i, blk in enumerate(self.block1_2):
            x_1_2 = blk(x_1_2, H_1, W_1)
        x_1_2 = self.norm1_2(x_1_2)
        x_1_2_reshape = x_1_2.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 64, 118, 118])
        # print("x_1_2_reshape.shape:",x_1_2_reshape.shape)   # torch.Size([2, 64, 128, 128])


        x_2_1 = fused_outputs[1]    # ([2, 128, 64, 64])
        x_2_1 = x_2_1.view(x_2_1.size(0), -1, x_2_1.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_1):
            x_2_1 = blk(x_2_1, H_2, W_2)
        x_2_1 = self.norm2_1(x_2_1)
        x_2_1_reshape = x_2_1.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        # print("x_2_1_reshape.shape:",x_2_1_reshape.shape)   # torch.Size([2, 128, 64, 64])

        x_3 = fused_outputs[2]    # ([2, 320, 32, 32]) 
        # print(x_3.shape)
        # import pdb; pdb.set_trace()
        x_3 = x_3.view(x_3.size(0), -1, x_3.size(1))    # torch.Size([2, 32*32, 320])
        for i, blk in enumerate(self.block3):
            x_3 = blk(x_3, H_3, W_3)
        x_3 = self.norm3(x_3)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        x_3_reshape = x_3.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 320, 32, 32]) 
        # outs.append(x_3_reshape)
        # print("stage3, x_3.shape:",x_3_reshape.shape)   # torch.Size([2, 320, 32, 32]) 


        
        # stage 4
        x_4, H_4, W_4 = self.patch_embed4(x_3_reshape)
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])

        x_4 = x_4.reshape(B, H_4, W_4, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 512, 15, 15])

        # Fuse x_1_2_reshape and x_2_1_reshape and x_3_reshape and x_4
        # torch.Size([2, 64, 128, 128])/torch.Size([2, 128, 64, 64])/torch.Size([2, 320, 32, 32]) / torch.Size([2, 512, 15, 15])
        fused_outputs_2 = self.fuse_layers2([x_1_2_reshape, x_2_1_reshape, x_3_reshape, x_4])


        x_1_3 = fused_outputs_2[0]
        # print("x_1_3.shape:",x_1_3.shape)  # torch.Size([2, 64, 128, 128])
        x_1_3 = x_1_3.view(x_1_3.size(0), -1, x_1_3.size(1))    
        for i, blk in enumerate(self.block1_3):
            x_1_3 = blk(x_1_3, H_1, W_1)
        x_1_3 = self.norm1_3(x_1_3)
        x_1_3_reshape = x_1_3.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_1_3_reshape.shape:",x_1_3_reshape.shape)   # torch.Size([2, 64, 128, 128])
        outs.append(x_1_3_reshape)


        x_2_2 = fused_outputs_2[1]
        # print(x_2_2.shape)  # torch.Size([2, 128, 64, 64])
        x_2_2 = x_2_2.view(x_2_2.size(0), -1, x_2_2.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_2):
            x_2_2 = blk(x_2_2, H_2, W_2)
        x_2_2 = self.norm2_2(x_2_2)
        x_2_2_reshape = x_2_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        # print("x_2_2_reshape.shape:",x_2_2_reshape.shape)   # torch.Size([2, 128, 64, 64])
        outs.append(x_2_2_reshape)

        x_3_1 = fused_outputs_2[2]
        # print(x_3_1.shape)  # torch.Size([2, 320, 32, 32])
        x_3_1 = x_3_1.view(x_3_1.size(0), -1, x_3_1.size(1))    # torch.Size([2, 32*32, 320])
        for i, blk in enumerate(self.block3_1):
            x_3_1 = blk(x_3_1, H_3, W_3)
        x_3_1 = self.norm3_1(x_3_1)
        x_3_1_reshape = x_3_1.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 320, 32, 32])
        # print("x_3_1_reshape.shape:",x_3_1_reshape.shape)   # torch.Size([2, 320, 32, 32])
        outs.append(x_3_1_reshape)


        x_4 = fused_outputs_2[3]    # torch.Size([2, 512, 15, 15])
        x_4 = x_4.view(x_4.size(0), -1, x_4.size(1))    # torch.Size([2, 15*15, 512])
        for i, blk in enumerate(self.block4):
            x_4 = blk(x_4, H_4, W_4)
        x_4 = self.norm4(x_4)
        x_4_reshape = x_4.reshape(B, H_4, W_4, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 512, 15, 15])
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])

        outs.append(x_4_reshape)
        # print("stage4, x_4.shape:",x_4.shape)   # torch.Size([2, 512, 15, 15])


        # import pdb;pdb.set_trace()
        return outs

    def forward_seg_hr_mamba_v3(self, x):
        # print("x.shape:",x.shape)   # torch.Size([2, 3, 512, 512])
        # x = F.interpolate(x, size=(472, 472), mode='bilinear', align_corners=False)
        # import pdb;pdb.set_trace()
        B = x.shape[0]
        outs = []

        # stage 1
        x_1, H_1, W_1 = self.patch_embed1(x)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        for i, blk in enumerate(self.block1):
            x_1 = blk(x_1, H_1, W_1)
        x_1 = self.norm1(x_1)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        x_1_reshape = x_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 64, 118, 118])
        outs.append(x_1_reshape)
        # print("stage1, x.shape:",x.shape)   #torch.Size([2, 64, 118, 118])

        # mamba block 1_1
        for i, blk in enumerate(self.block1_1):
            x_1_1 = blk(x_1, H_1, W_1)
        x_1_1 = self.norm1_1(x_1_1)
        # print("x_1_1.shape:",x_1_1.shape)   # torch.Size([2, 16384, 64])
        x_1_1_reshape = x_1_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_1_1_reshape.shape:",x_1_1_reshape.shape)   # torch.Size([2, 64, 128, 128])

        # stage 2
        x_2, H_2, W_2 = self.patch_embed2(x_1_reshape)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        for i, blk in enumerate(self.block2):
            x_2 = blk(x_2, H_2, W_2)
        x_2 = self.norm2(x_2)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        x_2_reshape = x_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 128, 59, 59])
        outs.append(x_2_reshape)
        # print("x_2_reshape.shape:",x_2_reshape.shape)   # torch.Size([2, 128, 64, 64])

        # Fuse x_1_1_reshape and x_2_reshape
        fused_outputs = self.fuse_layers1([x_1_1_reshape, x_2_reshape])

        # mamba block 1-2
        x_1_2 = fused_outputs[0]    # torch.Size([2, 64, 128, 128])
        x_1_2 = x_1_2.view(x_1_2.size(0), -1, x_1_2.size(1))    # torch.Size([2, 13924, 64])
        for i, blk in enumerate(self.block1_2):
            x_1_2 = blk(x_1_2, H_1, W_1)
        x_1_2 = self.norm1_2(x_1_2)
        x_1_2_reshape = x_1_2.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 64, 118, 118])
        # print("x_1_2_reshape.shape:",x_1_2_reshape.shape)   # torch.Size([2, 64, 128, 128])


        x_2_1 = fused_outputs[1]    # ([2, 128, 64, 64])
        x_2_1 = x_2_1.view(x_2_1.size(0), -1, x_2_1.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_1):
            x_2_1 = blk(x_2_1, H_2, W_2)
        x_2_1 = self.norm2_1(x_2_1)
        x_2_1_reshape = x_2_1.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        # print("x_2_1_reshape.shape:",x_2_1_reshape.shape)   # torch.Size([2, 128, 64, 64])




        # stage 3
        x_3, H_3, W_3 = self.patch_embed3(x_2_reshape)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        for i, blk in enumerate(self.block3):
            x_3 = blk(x_3, H_3, W_3)
        x_3 = self.norm3(x_3)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        x_3_reshape = x_3.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous()
        # print("x.shape:",x.shape)   # torch.Size([2, 320, 32, 32]) 
        outs.append(x_3_reshape)
        # print("stage3, x_3.shape:",x_3_reshape.shape)   # torch.Size([2, 320, 32, 32]) 


        # Fuse x_1_2_reshape and x_2_1_reshape and x_3_reshape
        # torch.Size([2, 64, 128, 128])/torch.Size([2, 128, 64, 64])/torch.Size([2, 320, 32, 32]) 
        fused_outputs = self.fuse_layers2([x_1_2_reshape, x_2_1_reshape, x_3_reshape])

        x_1_3 = fused_outputs[0]
        # print("x_1_3.shape:",x_1_3.shape)  # torch.Size([2, 64, 128, 128])
        x_1_3 = x_1_3.view(x_1_3.size(0), -1, x_1_3.size(1))    
        for i, blk in enumerate(self.block1_3):
            x_1_3 = blk(x_1_3, H_1, W_1)
        x_1_3 = self.norm1_3(x_1_3)
        x_1_3_reshape = x_1_3.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_1_3_reshape.shape:",x_1_3_reshape.shape)   # torch.Size([2, 64, 128, 128])
        outs.append(x_1_3_reshape)


        x_2_2 = fused_outputs[1]
        # print(x_2_2.shape)  # torch.Size([2, 128, 64, 64])
        x_2_2 = x_2_2.view(x_2_2.size(0), -1, x_2_2.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_2):
            x_2_2 = blk(x_2_2, H_2, W_2)
        x_2_2 = self.norm2_2(x_2_2)
        x_2_2_reshape = x_2_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        # print("x_2_2_reshape.shape:",x_2_2_reshape.shape)   # torch.Size([2, 128, 64, 64])
        outs.append(x_2_2_reshape)

        x_3_1 = fused_outputs[2]
        # print(x_3_1.shape)  # torch.Size([2, 320, 32, 32])
        x_3_1 = x_3_1.view(x_3_1.size(0), -1, x_3_1.size(1))    # torch.Size([2, 32*32, 320])
        for i, blk in enumerate(self.block3_1):
            x_3_1 = blk(x_3_1, H_3, W_3)
        x_3_1 = self.norm3_1(x_3_1)
        x_3_1_reshape = x_3_1.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 320, 32, 32])
        # print("x_3_1_reshape.shape:",x_3_1_reshape.shape)   # torch.Size([2, 320, 32, 32])
        outs.append(x_3_1_reshape)







        # stage 4
        x_4, H_4, W_4 = self.patch_embed4(x_3_reshape)
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])
        for i, blk in enumerate(self.block4):
            x_4 = blk(x_4, H_4, W_4)
        x_4 = self.norm4(x_4)
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])
        x_4 = x_4.reshape(B, H_4, W_4, -1).permute(0, 3, 1, 2).contiguous()
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 512, 15, 15])
        outs.append(x_4)
        # print("stage4, x_4.shape:",x_4.shape)   # torch.Size([2, 512, 15, 15])

        # import pdb;pdb.set_trace()
        return outs

        
    def forward_seg_hr_mamba_v4(self, x):
        # print("x.shape:",x.shape)   # torch.Size([2, 3, 512, 512])
        # x = F.interpolate(x, size=(472, 472), mode='bilinear', align_corners=False)
        # import pdb;pdb.set_trace()
        B = x.shape[0]
        outs = []

        # stage 1
        x_1, H_1, W_1 = self.patch_embed1(x)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        for i, blk in enumerate(self.block1):
            x_1 = blk(x_1, H_1, W_1)
        x_1 = self.norm1(x_1)
        # print("x_1.shape:",x_1.shape)   # torch.Size([2, 16384, 64])
        x_1_reshape = x_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        x_1_reshape = F.relu(x_1_reshape)  # 添加 ReLU 操作
        # print("x.shape:",x.shape)   # torch.Size([2, 64, 118, 118])
        # outs.append(x_1_reshape)
        # print("stage1, x.shape:",x.shape)   #torch.Size([2, 64, 118, 118])

        # mamba block 1_1
        for i, blk in enumerate(self.block1_1):
            x_1_1 = blk(x_1, H_1, W_1)
        x_1_1 = self.norm1_1(x_1_1)
        # print("x_1_1.shape:",x_1_1.shape)   # torch.Size([2, 16384, 64])
        x_1_1_reshape = x_1_1.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        x_1_1_reshape = F.relu(x_1_1_reshape)  # 添加 ReLU 操作
        # print("x_1_1_reshape.shape:",x_1_1_reshape.shape)   # torch.Size([2, 64, 128, 128])

        # stage 2
        x_2, H_2, W_2 = self.patch_embed2(x_1_reshape)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        for i, blk in enumerate(self.block2):
            x_2 = blk(x_2, H_2, W_2)
        x_2 = self.norm2(x_2)
        # print("x_2.shape:",x_2.shape)   # torch.Size([2, 4096, 128])
        x_2_reshape = x_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous()
        x_2_reshape = F.relu(x_2_reshape)  # 添加 ReLU 操作
        # print("x.shape:",x.shape)   # torch.Size([2, 128, 59, 59])
        # outs.append(x_2_reshape)
        # print("x_2_reshape.shape:",x_2_reshape.shape)   # torch.Size([2, 128, 64, 64])

        # Fuse x_1_1_reshape and x_2_reshape
        fused_outputs = self.fuse_layers1([x_1_1_reshape, x_2_reshape])

        # mamba block 1-2
        x_1_2 = fused_outputs[0]    # torch.Size([2, 64, 128, 128])
        x_1_2 = x_1_2.view(x_1_2.size(0), -1, x_1_2.size(1))    # torch.Size([2, 13924, 64])
        for i, blk in enumerate(self.block1_2):
            x_1_2 = blk(x_1_2, H_1, W_1)
        x_1_2 = self.norm1_2(x_1_2)
        x_1_2_reshape = x_1_2.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 64, 118, 118])
        x_1_2_reshape = F.relu(x_1_2_reshape)  # 添加 ReLU 操作
        # print("x_1_2_reshape.shape:",x_1_2_reshape.shape)   # torch.Size([2, 64, 128, 128])


        x_2_1 = fused_outputs[1]    # ([2, 128, 64, 64])
        x_2_1 = x_2_1.view(x_2_1.size(0), -1, x_2_1.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_1):
            x_2_1 = blk(x_2_1, H_2, W_2)
        x_2_1 = self.norm2_1(x_2_1)
        x_2_1_reshape = x_2_1.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        x_2_1_reshape = F.relu(x_2_1_reshape)  # 添加 ReLU 操作
        # print("x_2_1_reshape.shape:",x_2_1_reshape.shape)   # torch.Size([2, 128, 64, 64])




        # stage 3
        x_3, H_3, W_3 = self.patch_embed3(x_2_reshape)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        for i, blk in enumerate(self.block3):
            x_3 = blk(x_3, H_3, W_3)
        x_3 = self.norm3(x_3)
        # print("x_3.shape:",x_3.shape)   # torch.Size([2, 1024, 320])
        x_3_reshape = x_3.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous()
        x_3_reshape = F.relu(x_3_reshape)  # 添加 ReLU 操作
        # print("x.shape:",x.shape)   # torch.Size([2, 320, 32, 32]) 
        # outs.append(x_3_reshape)
        # print("stage3, x_3.shape:",x_3_reshape.shape)   # torch.Size([2, 320, 32, 32]) 


        # Fuse x_1_2_reshape and x_2_1_reshape and x_3_reshape
        # torch.Size([2, 64, 128, 128])/torch.Size([2, 128, 64, 64])/torch.Size([2, 320, 32, 32]) 
        fused_outputs = self.fuse_layers2([x_1_2_reshape, x_2_1_reshape, x_3_reshape])

        x_1_3 = fused_outputs[0]
        # print("x_1_3.shape:",x_1_3.shape)  # torch.Size([2, 64, 128, 128])
        x_1_3 = x_1_3.view(x_1_3.size(0), -1, x_1_3.size(1))    
        for i, blk in enumerate(self.block1_3):
            x_1_3 = blk(x_1_3, H_1, W_1)
        x_1_3 = self.norm1_3(x_1_3)
        x_1_3_reshape = x_1_3.reshape(B, H_1, W_1, -1).permute(0, 3, 1, 2).contiguous()
        x_1_3_reshape = F.relu(x_1_3_reshape)  # 添加 ReLU 操作
        # print("x_1_3_reshape.shape:",x_1_3_reshape.shape)   # torch.Size([2, 64, 128, 128])
        outs.append(x_1_3_reshape)


        x_2_2 = fused_outputs[1]
        # print(x_2_2.shape)  # torch.Size([2, 128, 64, 64])
        x_2_2 = x_2_2.view(x_2_2.size(0), -1, x_2_2.size(1))    # torch.Size([2, 3481, 128])
        for i, blk in enumerate(self.block2_2):
            x_2_2 = blk(x_2_2, H_2, W_2)
        x_2_2 = self.norm2_2(x_2_2)
        x_2_2_reshape = x_2_2.reshape(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 128, 64, 64])
        x_2_2_reshape = F.relu(x_2_2_reshape)  # 添加 ReLU 操作
        # print("x_2_2_reshape.shape:",x_2_2_reshape.shape)   # torch.Size([2, 128, 64, 64])
        outs.append(x_2_2_reshape)

        x_3_1 = fused_outputs[2]
        # print(x_3_1.shape)  # torch.Size([2, 320, 32, 32])
        x_3_1 = x_3_1.view(x_3_1.size(0), -1, x_3_1.size(1))    # torch.Size([2, 32*32, 320])
        for i, blk in enumerate(self.block3_1):
            x_3_1 = blk(x_3_1, H_3, W_3)
        x_3_1 = self.norm3_1(x_3_1)
        x_3_1_reshape = x_3_1.reshape(B, H_3, W_3, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([2, 320, 32, 32])
        x_3_1_reshape = F.relu(x_3_1_reshape)  # 添加 ReLU 操作
        # print("x_3_1_reshape.shape:",x_3_1_reshape.shape)   # torch.Size([2, 320, 32, 32])
        outs.append(x_3_1_reshape)







        # stage 4
        x_4, H_4, W_4 = self.patch_embed4(x_3_reshape)
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])
        for i, blk in enumerate(self.block4):
            x_4 = blk(x_4, H_4, W_4)
        x_4 = self.norm4(x_4)
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 225, 512])
        x_4 = x_4.reshape(B, H_4, W_4, -1).permute(0, 3, 1, 2).contiguous()
        x_4 = F.relu(x_4)  # 添加 ReLU 操作
        # print("x_4.shape:",x_4.shape)   # torch.Size([2, 512, 15, 15])
        outs.append(x_4)
        # print("stage4, x_4.shape:",x_4.shape)   # torch.Size([2, 512, 15, 15])

        # import pdb;pdb.set_trace()
        return outs

   

    def forward(self, x):
        # x = self.forward_features(x)  # segMamba_v1:将transformer block 替换成 mamba block
        # x = self.forward_mamba(x)
        # x = self.forward_mamba_v4(x)
        # x = self.forward_seg_hr_mamba(x)    # 在segMamba_v1的基础上，加上hrformer高低分辨率特征融合的思想
        # x = self.forward_seg_hr_mamba_v2(x)    # 在seg_hr_mamba的基础上，调整block3、block4的特征融合 效果不如forward_seg_hr_mamba
        # x = self.forward_seg_hr_mamba_v3(x)    # 在forward_seg_hr_mamba的基础上，加上block1、2、3的特征图输出
        x = self.forward_seg_hr_mamba_4(x)    # 在forward_seg_hr_mamba的基础上,调整特征融合操作、添加relu（直接将relu操作后的特征图放入解码器）、
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



@BACKBONES.register_module()
class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)