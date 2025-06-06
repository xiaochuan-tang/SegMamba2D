# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

import mmseg.Vim.vim.models_mamba as models_mamba

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # self.linear_w4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        # self.linear_w3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        # self.linear_w2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        # self.linear_w1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,  # seg_hr_mamba 需要额外*2
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.mamba_block = None

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    # segformer forward
    def forward(self, inputs):
        # print("inputs:",inputs)
        # print("inputs[0].shape:",inputs[0].shape)   # torch.Size([2, 64, 99, 99])
        # print("inputs[1].shape:",inputs[1].shape)   # torch.Size([2, 128, 50, 50])
        # print("inputs[2].shape:",inputs[2].shape)   # torch.Size([2, 320, 25, 25])
        # print("inputs[3].shape:",inputs[3].shape)   # torch.Size([2, 512, 13, 13])
        # import pdb;pdb.set_trace()

        x = self._transform_inputs(inputs)  # len=4, H,W: 1/4,1/8,1/16,1/32
        # print("x1:",x.shape)
        c1, c2, c3, c4 = x

        # print("c1.shape:",c1.shape)   # torch.Size([2, 64, 99, 99])
        # print("c2.shape:",c2.shape)   # torch.Size([2, 128, 50, 50])
        # print("c3.shape:",c3.shape)   # torch.Size([2, 320, 25, 25])
        # print("c4.shape:",c4.shape)   # torch.Size([2, 512, 13, 13])
        # import pdb;pdb.set_trace()

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        # print("_c4.shape:",_c4.shape)   # torch.Size([2, 768, 99, 99])

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        # print("_c3.shape:",_c3.shape)   # torch.Size([2, 768, 99, 99])

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        # print("_c2.shape:",_c2.shape)   # torch.Size([2, 768, 99, 99])

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # print("_c1.shape:",_c1.shape)   # torch.Size([2, 768, 99, 99])

        c_cat = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        # print(c_cat.shape)  # torch.Size([2, 3072, 99, 99])

        # B, C, H, W = c_cat.shape
        # c_cat = c_cat.view(B, C, H * W).permute(0, 2, 1)

        # if self.mamba_block is None:
        #     self.mamba_block = models_mamba.create_block(d_model=C).to("cuda")
        # hidden_states, residual = self.mamba_block(c_cat)
        # hidden_states = hidden_states.permute(0, 2, 1).view(B, C, H, W)
        # _c = self.linear_fuse(hidden_states)
        
        _c = self.linear_fuse(c_cat)

        
        # print("_c.shape:",_c.shape)   # torch.Size([2, 768, 99, 99])

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print("x.shape:",x.shape)   # torch.Size([2, 2, 99, 99])

        # print("x2:",x.shape)
        # import pdb;pdb.set_trace()

        return x

    # # seg_hr_mamba forward
    # def forward(self, inputs):
    #     # print("inputs:",inputs)
    #     outs_1 = []
    #     outs_1.extend([inputs[0], inputs[1], inputs[2], inputs[6]])
    #     outs_2 = []
    #     outs_2.extend([inputs[3], inputs[4], inputs[5], inputs[6]])

    #     # print("inputs[0].shape:",inputs[0].shape)   # torch.Size([2, 64, 128, 128])
    #     # print("inputs[1].shape:",inputs[1].shape)   # torch.Size([2, 128, 64, 64])
    #     # print("inputs[2].shape:",inputs[2].shape)   # torch.Size([2, 320, 32, 32])
    #     # print("inputs[3].shape:",inputs[3].shape)   # torch.Size([2, 64, 128, 128])
    #     # print("inputs[4].shape:",inputs[4].shape)   # torch.Size([2, 128, 64, 64])
    #     # print("inputs[5].shape:",inputs[5].shape)   # torch.Size([2, 320, 32, 32])
    #     # print("inputs[6].shape:",inputs[6].shape)   # torch.Size([2, 512, 16, 16])

    #     # import pdb;pdb.set_trace()

    #     x1 = self._transform_inputs(outs_1)  # len=4, H,W: 1/4,1/8,1/16,1/32
    #     x2 = self._transform_inputs(outs_2)  # len=4, H,W: 1/4,1/8,1/16,1/32
    #     # print("x1:",x.shape)
    #     c1, c2, c3, c4 = x1
    #     w1, w2, w3, w4 = x2

    #     # print("c1.shape:",c1.shape)   # torch.Size([2, 64, 99, 99])
    #     # print("c2.shape:",c2.shape)   # torch.Size([2, 128, 50, 50])
    #     # print("c3.shape:",c3.shape)   # torch.Size([2, 320, 25, 25])
    #     # print("c4.shape:",c4.shape)   # torch.Size([2, 512, 13, 13])
    #     # print("c5.shape:",c5.shape)   # torch.Size([2, 128, 50, 50])
    #     # print("c6.shape:",c6.shape)   # torch.Size([2, 320, 25, 25])
    #     # print("c7.shape:",c7.shape)   # torch.Size([2, 512, 13, 13])
    #     # import pdb;pdb.set_trace()

    #     ############## MLP decoder on C1-C4 ###########
    #     n, _, h, w = c4.shape

    #     _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
    #     _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
    #     # print("_c4.shape:",_c4.shape)   # torch.Size([2, 768, 99, 99])

    #     _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
    #     _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
    #     # print("_c3.shape:",_c3.shape)   # torch.Size([2, 768, 99, 99])

    #     _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
    #     _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
    #     # print("_c2.shape:",_c2.shape)   # torch.Size([2, 768, 99, 99])

    #     _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
    #     # print("_c1.shape:",_c1.shape)   # torch.Size([2, 768, 99, 99])


    #     _w4 = self.linear_w4(w4).permute(0,2,1).reshape(n, -1, w4.shape[2], w4.shape[3])
    #     _w4 = resize(_w4, size=w1.size()[2:],mode='bilinear',align_corners=False)
    #     # print("_c4.shape:",_c4.shape)   # torch.Size([2, 768, 99, 99])

    #     _w3 = self.linear_w3(w3).permute(0,2,1).reshape(n, -1, w3.shape[2], w3.shape[3])
    #     _w3 = resize(_w3, size=w1.size()[2:],mode='bilinear',align_corners=False)
    #     # print("_c3.shape:",_c3.shape)   # torch.Size([2, 768, 99, 99])

    #     _w2 = self.linear_w2(w2).permute(0,2,1).reshape(n, -1, w2.shape[2], w2.shape[3])
    #     _w2 = resize(_w2, size=w1.size()[2:],mode='bilinear',align_corners=False)
    #     # print("_c2.shape:",_c2.shape)   # torch.Size([2, 768, 99, 99])

    #     _w1 = self.linear_w1(w1).permute(0,2,1).reshape(n, -1, w1.shape[2], w1.shape[3])
    #     # print("_c1.shape:",_c1.shape)   # torch.Size([2, 768, 99, 99])

    #     c_cat = torch.cat([_c4, _c3, _c2, _c1], dim=1)
    #     w_cat = torch.cat([_w4, _w3, _w2, _w1], dim=1)
    #     cat = torch.cat([c_cat, w_cat], dim=1)
    #     # print(c_cat.shape)  # torch.Size([2, 3072, 99, 99])

    #     # B, C, H, W = c_cat.shape
    #     # c_cat = c_cat.view(B, C, H * W).permute(0, 2, 1)

    #     # if self.mamba_block is None:
    #     #     self.mamba_block = models_mamba.create_block(d_model=C).to("cuda")
    #     # hidden_states, residual = self.mamba_block(c_cat)
    #     # hidden_states = hidden_states.permute(0, 2, 1).view(B, C, H, W)
    #     # _c = self.linear_fuse(hidden_states)
        
    #     _c = self.linear_fuse(cat)

        
    #     # print("_c.shape:",_c.shape)   # torch.Size([2, 768, 99, 99])

    #     x = self.dropout(_c)
    #     x = self.linear_pred(x)
    #     # print("x.shape:",x.shape)   # torch.Size([2, 2, 99, 99])

    #     # print("x2:",x.shape)
    #     # import pdb;pdb.set_trace()

    #     return x
