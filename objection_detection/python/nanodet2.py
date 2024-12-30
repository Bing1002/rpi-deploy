import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
try:
    import torchvision
except:
    pass
import math 
import cv2 
import ncnn



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convbn2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.backbone_conv1_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.backbone_maxpool = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(3,3), padding=(1,1), return_indices=False, stride=(2,2))
        self.convbn2d_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=24, in_channels=24, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.convbn2d_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_0_branch1_4 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_0_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_4 = nn.Conv2d(bias=True, dilation=(1,1), groups=88, in_channels=88, kernel_size=(3,3), out_channels=88, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.convbn2d_5 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=88, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_0_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_0 = nn.ChannelShuffle(groups=2)
        self.convbn2d_6 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=88, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_1_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_7 = nn.Conv2d(bias=True, dilation=(1,1), groups=88, in_channels=88, kernel_size=(3,3), out_channels=88, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_8 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=88, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_1_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_1 = nn.ChannelShuffle(groups=2)
        self.convbn2d_9 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=88, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_2_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_10 = nn.Conv2d(bias=True, dilation=(1,1), groups=88, in_channels=88, kernel_size=(3,3), out_channels=88, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_11 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=88, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_2_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_2 = nn.ChannelShuffle(groups=2)
        self.convbn2d_12 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=88, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_3_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_13 = nn.Conv2d(bias=True, dilation=(1,1), groups=88, in_channels=88, kernel_size=(3,3), out_channels=88, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_14 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=88, kernel_size=(1,1), out_channels=88, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage2_3_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_3 = nn.ChannelShuffle(groups=2)
        self.convbn2d_15 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.convbn2d_16 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_0_branch1_4 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_17 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_0_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_18 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.convbn2d_19 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_0_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_4 = nn.ChannelShuffle(groups=2)
        self.convbn2d_20 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_1_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_21 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_22 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_1_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_5 = nn.ChannelShuffle(groups=2)
        self.convbn2d_23 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_2_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_24 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_25 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_2_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_6 = nn.ChannelShuffle(groups=2)
        self.convbn2d_26 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_3_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_27 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_28 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_3_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_7 = nn.ChannelShuffle(groups=2)
        self.convbn2d_29 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_4_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_30 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_31 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_4_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_8 = nn.ChannelShuffle(groups=2)
        self.convbn2d_32 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_5_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_33 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_34 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_5_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_9 = nn.ChannelShuffle(groups=2)
        self.convbn2d_35 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_6_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_36 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_37 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_6_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_10 = nn.ChannelShuffle(groups=2)
        self.convbn2d_38 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_7_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_39 = nn.Conv2d(bias=True, dilation=(1,1), groups=176, in_channels=176, kernel_size=(3,3), out_channels=176, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_40 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=176, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage3_7_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_11 = nn.ChannelShuffle(groups=2)
        self.convbn2d_41 = nn.Conv2d(bias=True, dilation=(1,1), groups=352, in_channels=352, kernel_size=(3,3), out_channels=352, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.convbn2d_42 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_0_branch1_4 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_43 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_0_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_44 = nn.Conv2d(bias=True, dilation=(1,1), groups=352, in_channels=352, kernel_size=(3,3), out_channels=352, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.convbn2d_45 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_0_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_12 = nn.ChannelShuffle(groups=2)
        self.convbn2d_46 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_1_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_47 = nn.Conv2d(bias=True, dilation=(1,1), groups=352, in_channels=352, kernel_size=(3,3), out_channels=352, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_48 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_1_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_13 = nn.ChannelShuffle(groups=2)
        self.convbn2d_49 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_2_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_50 = nn.Conv2d(bias=True, dilation=(1,1), groups=352, in_channels=352, kernel_size=(3,3), out_channels=352, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_51 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_2_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_14 = nn.ChannelShuffle(groups=2)
        self.convbn2d_52 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_3_branch2_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_53 = nn.Conv2d(bias=True, dilation=(1,1), groups=352, in_channels=352, kernel_size=(3,3), out_channels=352, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_54 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=352, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.backbone_stage4_3_branch2_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.channelshuffle_15 = nn.ChannelShuffle(groups=2)
        self.convbn2d_55 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=176, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.fpn_reduce_layers_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_56 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=352, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.fpn_reduce_layers_1_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_57 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=704, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.fpn_reduce_layers_2_act = nn.LeakyReLU(negative_slope=0.100000)
        self.fpn_upsample = nn.Upsample(align_corners=False, mode='bilinear', scale_factor=(2.000000,2.000000), size=None)
        self.convbn2d_58 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.fpn_top_down_blocks_0_blocks_0_ghost1_primary_conv_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_59 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.fpn_top_down_blocks_0_blocks_0_ghost1_cheap_operation_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_60 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_61 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_62 = nn.Conv2d(bias=True, dilation=(1,1), groups=256, in_channels=256, kernel_size=(5,5), out_channels=256, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.convbn2d_63 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_0 = nn.Upsample(align_corners=False, mode='bilinear', scale_factor=(2.000000,2.000000), size=None)
        self.convbn2d_64 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.fpn_top_down_blocks_1_blocks_0_ghost1_primary_conv_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_65 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.fpn_top_down_blocks_1_blocks_0_ghost1_cheap_operation_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_66 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_67 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_68 = nn.Conv2d(bias=True, dilation=(1,1), groups=256, in_channels=256, kernel_size=(5,5), out_channels=256, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.convbn2d_69 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_70 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(2,2))
        self.fpn_downsamples_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_71 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_1 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_72 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.fpn_bottom_up_blocks_0_blocks_0_ghost1_primary_conv_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_73 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.fpn_bottom_up_blocks_0_blocks_0_ghost1_cheap_operation_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_74 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_75 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_76 = nn.Conv2d(bias=True, dilation=(1,1), groups=256, in_channels=256, kernel_size=(5,5), out_channels=256, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.convbn2d_77 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_78 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(2,2))
        self.fpn_downsamples_1_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_79 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_80 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.fpn_bottom_up_blocks_1_blocks_0_ghost1_primary_conv_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_81 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.fpn_bottom_up_blocks_1_blocks_0_ghost1_cheap_operation_2 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_82 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_83 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.convbn2d_84 = nn.Conv2d(bias=True, dilation=(1,1), groups=256, in_channels=256, kernel_size=(5,5), out_channels=256, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.convbn2d_85 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_86 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(2,2))
        self.fpn_extra_lvl_in_conv_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_87 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_3 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_88 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(2,2))
        self.fpn_extra_lvl_out_conv_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_89 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_4 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_90 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_0_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_91 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_5 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_92 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_0_1_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_93 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_6 = nn.LeakyReLU(negative_slope=0.100000)
        self.head_gfl_cls_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=112, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_94 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_1_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_95 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_7 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_96 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_1_1_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_97 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_8 = nn.LeakyReLU(negative_slope=0.100000)
        self.head_gfl_cls_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=112, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_98 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_2_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_99 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_9 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_100 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_2_1_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_101 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_10 = nn.LeakyReLU(negative_slope=0.100000)
        self.head_gfl_cls_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=112, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.convbn2d_102 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_3_0_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_103 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_11 = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_104 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(5,5), out_channels=128, padding=(2,2), padding_mode='zeros', stride=(1,1))
        self.head_cls_convs_3_1_act = nn.LeakyReLU(negative_slope=0.100000)
        self.convbn2d_105 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.pnnx_unique_12 = nn.LeakyReLU(negative_slope=0.100000)
        self.head_gfl_cls_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=112, padding=(0,0), padding_mode='zeros', stride=(1,1))

        archive = zipfile.ZipFile('/home/bing/code/checkpoints/nanodet/nanodet_plus_m_1.5x_416_torchscript.pnnx.bin', 'r')
        self.convbn2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.bias', (24), 'float32')
        self.convbn2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.weight', (24,3,3,3), 'float32')
        self.convbn2d_1.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.bias', (24), 'float32')
        self.convbn2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.weight', (24,1,3,3), 'float32')
        self.convbn2d_2.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.bias', (88), 'float32')
        self.convbn2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.weight', (88,24,1,1), 'float32')
        self.convbn2d_3.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.bias', (88), 'float32')
        self.convbn2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.weight', (88,24,1,1), 'float32')
        self.convbn2d_4.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.bias', (88), 'float32')
        self.convbn2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.weight', (88,1,3,3), 'float32')
        self.convbn2d_5.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_5.bias', (88), 'float32')
        self.convbn2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_5.weight', (88,88,1,1), 'float32')
        self.convbn2d_6.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_6.bias', (88), 'float32')
        self.convbn2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_6.weight', (88,88,1,1), 'float32')
        self.convbn2d_7.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_7.bias', (88), 'float32')
        self.convbn2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_7.weight', (88,1,3,3), 'float32')
        self.convbn2d_8.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_8.bias', (88), 'float32')
        self.convbn2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_8.weight', (88,88,1,1), 'float32')
        self.convbn2d_9.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_9.bias', (88), 'float32')
        self.convbn2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_9.weight', (88,88,1,1), 'float32')
        self.convbn2d_10.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_10.bias', (88), 'float32')
        self.convbn2d_10.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_10.weight', (88,1,3,3), 'float32')
        self.convbn2d_11.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_11.bias', (88), 'float32')
        self.convbn2d_11.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_11.weight', (88,88,1,1), 'float32')
        self.convbn2d_12.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_12.bias', (88), 'float32')
        self.convbn2d_12.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_12.weight', (88,88,1,1), 'float32')
        self.convbn2d_13.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_13.bias', (88), 'float32')
        self.convbn2d_13.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_13.weight', (88,1,3,3), 'float32')
        self.convbn2d_14.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_14.bias', (88), 'float32')
        self.convbn2d_14.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_14.weight', (88,88,1,1), 'float32')
        self.convbn2d_15.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_15.bias', (176), 'float32')
        self.convbn2d_15.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_15.weight', (176,1,3,3), 'float32')
        self.convbn2d_16.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_16.bias', (176), 'float32')
        self.convbn2d_16.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_16.weight', (176,176,1,1), 'float32')
        self.convbn2d_17.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_17.bias', (176), 'float32')
        self.convbn2d_17.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_17.weight', (176,176,1,1), 'float32')
        self.convbn2d_18.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_18.bias', (176), 'float32')
        self.convbn2d_18.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_18.weight', (176,1,3,3), 'float32')
        self.convbn2d_19.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_19.bias', (176), 'float32')
        self.convbn2d_19.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_19.weight', (176,176,1,1), 'float32')
        self.convbn2d_20.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_20.bias', (176), 'float32')
        self.convbn2d_20.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_20.weight', (176,176,1,1), 'float32')
        self.convbn2d_21.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_21.bias', (176), 'float32')
        self.convbn2d_21.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_21.weight', (176,1,3,3), 'float32')
        self.convbn2d_22.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_22.bias', (176), 'float32')
        self.convbn2d_22.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_22.weight', (176,176,1,1), 'float32')
        self.convbn2d_23.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_23.bias', (176), 'float32')
        self.convbn2d_23.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_23.weight', (176,176,1,1), 'float32')
        self.convbn2d_24.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_24.bias', (176), 'float32')
        self.convbn2d_24.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_24.weight', (176,1,3,3), 'float32')
        self.convbn2d_25.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_25.bias', (176), 'float32')
        self.convbn2d_25.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_25.weight', (176,176,1,1), 'float32')
        self.convbn2d_26.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_26.bias', (176), 'float32')
        self.convbn2d_26.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_26.weight', (176,176,1,1), 'float32')
        self.convbn2d_27.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_27.bias', (176), 'float32')
        self.convbn2d_27.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_27.weight', (176,1,3,3), 'float32')
        self.convbn2d_28.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_28.bias', (176), 'float32')
        self.convbn2d_28.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_28.weight', (176,176,1,1), 'float32')
        self.convbn2d_29.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_29.bias', (176), 'float32')
        self.convbn2d_29.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_29.weight', (176,176,1,1), 'float32')
        self.convbn2d_30.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_30.bias', (176), 'float32')
        self.convbn2d_30.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_30.weight', (176,1,3,3), 'float32')
        self.convbn2d_31.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_31.bias', (176), 'float32')
        self.convbn2d_31.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_31.weight', (176,176,1,1), 'float32')
        self.convbn2d_32.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_32.bias', (176), 'float32')
        self.convbn2d_32.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_32.weight', (176,176,1,1), 'float32')
        self.convbn2d_33.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_33.bias', (176), 'float32')
        self.convbn2d_33.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_33.weight', (176,1,3,3), 'float32')
        self.convbn2d_34.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_34.bias', (176), 'float32')
        self.convbn2d_34.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_34.weight', (176,176,1,1), 'float32')
        self.convbn2d_35.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_35.bias', (176), 'float32')
        self.convbn2d_35.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_35.weight', (176,176,1,1), 'float32')
        self.convbn2d_36.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_36.bias', (176), 'float32')
        self.convbn2d_36.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_36.weight', (176,1,3,3), 'float32')
        self.convbn2d_37.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_37.bias', (176), 'float32')
        self.convbn2d_37.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_37.weight', (176,176,1,1), 'float32')
        self.convbn2d_38.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_38.bias', (176), 'float32')
        self.convbn2d_38.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_38.weight', (176,176,1,1), 'float32')
        self.convbn2d_39.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_39.bias', (176), 'float32')
        self.convbn2d_39.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_39.weight', (176,1,3,3), 'float32')
        self.convbn2d_40.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_40.bias', (176), 'float32')
        self.convbn2d_40.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_40.weight', (176,176,1,1), 'float32')
        self.convbn2d_41.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_41.bias', (352), 'float32')
        self.convbn2d_41.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_41.weight', (352,1,3,3), 'float32')
        self.convbn2d_42.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_42.bias', (352), 'float32')
        self.convbn2d_42.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_42.weight', (352,352,1,1), 'float32')
        self.convbn2d_43.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_43.bias', (352), 'float32')
        self.convbn2d_43.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_43.weight', (352,352,1,1), 'float32')
        self.convbn2d_44.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_44.bias', (352), 'float32')
        self.convbn2d_44.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_44.weight', (352,1,3,3), 'float32')
        self.convbn2d_45.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_45.bias', (352), 'float32')
        self.convbn2d_45.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_45.weight', (352,352,1,1), 'float32')
        self.convbn2d_46.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_46.bias', (352), 'float32')
        self.convbn2d_46.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_46.weight', (352,352,1,1), 'float32')
        self.convbn2d_47.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_47.bias', (352), 'float32')
        self.convbn2d_47.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_47.weight', (352,1,3,3), 'float32')
        self.convbn2d_48.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_48.bias', (352), 'float32')
        self.convbn2d_48.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_48.weight', (352,352,1,1), 'float32')
        self.convbn2d_49.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_49.bias', (352), 'float32')
        self.convbn2d_49.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_49.weight', (352,352,1,1), 'float32')
        self.convbn2d_50.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_50.bias', (352), 'float32')
        self.convbn2d_50.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_50.weight', (352,1,3,3), 'float32')
        self.convbn2d_51.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_51.bias', (352), 'float32')
        self.convbn2d_51.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_51.weight', (352,352,1,1), 'float32')
        self.convbn2d_52.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_52.bias', (352), 'float32')
        self.convbn2d_52.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_52.weight', (352,352,1,1), 'float32')
        self.convbn2d_53.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_53.bias', (352), 'float32')
        self.convbn2d_53.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_53.weight', (352,1,3,3), 'float32')
        self.convbn2d_54.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_54.bias', (352), 'float32')
        self.convbn2d_54.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_54.weight', (352,352,1,1), 'float32')
        self.convbn2d_55.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_55.bias', (128), 'float32')
        self.convbn2d_55.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_55.weight', (128,176,1,1), 'float32')
        self.convbn2d_56.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_56.bias', (128), 'float32')
        self.convbn2d_56.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_56.weight', (128,352,1,1), 'float32')
        self.convbn2d_57.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_57.bias', (128), 'float32')
        self.convbn2d_57.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_57.weight', (128,704,1,1), 'float32')
        self.convbn2d_58.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_58.bias', (64), 'float32')
        self.convbn2d_58.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_58.weight', (64,256,1,1), 'float32')
        self.convbn2d_59.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_59.bias', (64), 'float32')
        self.convbn2d_59.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_59.weight', (64,1,3,3), 'float32')
        self.convbn2d_60.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_60.bias', (64), 'float32')
        self.convbn2d_60.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_60.weight', (64,128,1,1), 'float32')
        self.convbn2d_61.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_61.bias', (64), 'float32')
        self.convbn2d_61.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_61.weight', (64,1,3,3), 'float32')
        self.convbn2d_62.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_62.bias', (256), 'float32')
        self.convbn2d_62.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_62.weight', (256,1,5,5), 'float32')
        self.convbn2d_63.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_63.bias', (128), 'float32')
        self.convbn2d_63.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_63.weight', (128,256,1,1), 'float32')
        self.convbn2d_64.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_64.bias', (64), 'float32')
        self.convbn2d_64.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_64.weight', (64,256,1,1), 'float32')
        self.convbn2d_65.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_65.bias', (64), 'float32')
        self.convbn2d_65.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_65.weight', (64,1,3,3), 'float32')
        self.convbn2d_66.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_66.bias', (64), 'float32')
        self.convbn2d_66.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_66.weight', (64,128,1,1), 'float32')
        self.convbn2d_67.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_67.bias', (64), 'float32')
        self.convbn2d_67.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_67.weight', (64,1,3,3), 'float32')
        self.convbn2d_68.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_68.bias', (256), 'float32')
        self.convbn2d_68.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_68.weight', (256,1,5,5), 'float32')
        self.convbn2d_69.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_69.bias', (128), 'float32')
        self.convbn2d_69.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_69.weight', (128,256,1,1), 'float32')
        self.convbn2d_70.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_70.bias', (128), 'float32')
        self.convbn2d_70.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_70.weight', (128,1,5,5), 'float32')
        self.convbn2d_71.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_71.bias', (128), 'float32')
        self.convbn2d_71.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_71.weight', (128,128,1,1), 'float32')
        self.convbn2d_72.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_72.bias', (64), 'float32')
        self.convbn2d_72.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_72.weight', (64,256,1,1), 'float32')
        self.convbn2d_73.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_73.bias', (64), 'float32')
        self.convbn2d_73.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_73.weight', (64,1,3,3), 'float32')
        self.convbn2d_74.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_74.bias', (64), 'float32')
        self.convbn2d_74.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_74.weight', (64,128,1,1), 'float32')
        self.convbn2d_75.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_75.bias', (64), 'float32')
        self.convbn2d_75.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_75.weight', (64,1,3,3), 'float32')
        self.convbn2d_76.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_76.bias', (256), 'float32')
        self.convbn2d_76.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_76.weight', (256,1,5,5), 'float32')
        self.convbn2d_77.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_77.bias', (128), 'float32')
        self.convbn2d_77.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_77.weight', (128,256,1,1), 'float32')
        self.convbn2d_78.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_78.bias', (128), 'float32')
        self.convbn2d_78.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_78.weight', (128,1,5,5), 'float32')
        self.convbn2d_79.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_79.bias', (128), 'float32')
        self.convbn2d_79.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_79.weight', (128,128,1,1), 'float32')
        self.convbn2d_80.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_80.bias', (64), 'float32')
        self.convbn2d_80.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_80.weight', (64,256,1,1), 'float32')
        self.convbn2d_81.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_81.bias', (64), 'float32')
        self.convbn2d_81.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_81.weight', (64,1,3,3), 'float32')
        self.convbn2d_82.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_82.bias', (64), 'float32')
        self.convbn2d_82.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_82.weight', (64,128,1,1), 'float32')
        self.convbn2d_83.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_83.bias', (64), 'float32')
        self.convbn2d_83.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_83.weight', (64,1,3,3), 'float32')
        self.convbn2d_84.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_84.bias', (256), 'float32')
        self.convbn2d_84.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_84.weight', (256,1,5,5), 'float32')
        self.convbn2d_85.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_85.bias', (128), 'float32')
        self.convbn2d_85.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_85.weight', (128,256,1,1), 'float32')
        self.convbn2d_86.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_86.bias', (128), 'float32')
        self.convbn2d_86.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_86.weight', (128,1,5,5), 'float32')
        self.convbn2d_87.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_87.bias', (128), 'float32')
        self.convbn2d_87.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_87.weight', (128,128,1,1), 'float32')
        self.convbn2d_88.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_88.bias', (128), 'float32')
        self.convbn2d_88.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_88.weight', (128,1,5,5), 'float32')
        self.convbn2d_89.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_89.bias', (128), 'float32')
        self.convbn2d_89.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_89.weight', (128,128,1,1), 'float32')
        self.convbn2d_90.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_90.bias', (128), 'float32')
        self.convbn2d_90.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_90.weight', (128,1,5,5), 'float32')
        self.convbn2d_91.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_91.bias', (128), 'float32')
        self.convbn2d_91.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_91.weight', (128,128,1,1), 'float32')
        self.convbn2d_92.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_92.bias', (128), 'float32')
        self.convbn2d_92.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_92.weight', (128,1,5,5), 'float32')
        self.convbn2d_93.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_93.bias', (128), 'float32')
        self.convbn2d_93.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_93.weight', (128,128,1,1), 'float32')
        self.head_gfl_cls_0.bias = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.0.bias', (112), 'float32')
        self.head_gfl_cls_0.weight = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.0.weight', (112,128,1,1), 'float32')
        self.convbn2d_94.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_94.bias', (128), 'float32')
        self.convbn2d_94.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_94.weight', (128,1,5,5), 'float32')
        self.convbn2d_95.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_95.bias', (128), 'float32')
        self.convbn2d_95.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_95.weight', (128,128,1,1), 'float32')
        self.convbn2d_96.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_96.bias', (128), 'float32')
        self.convbn2d_96.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_96.weight', (128,1,5,5), 'float32')
        self.convbn2d_97.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_97.bias', (128), 'float32')
        self.convbn2d_97.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_97.weight', (128,128,1,1), 'float32')
        self.head_gfl_cls_1.bias = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.1.bias', (112), 'float32')
        self.head_gfl_cls_1.weight = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.1.weight', (112,128,1,1), 'float32')
        self.convbn2d_98.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_98.bias', (128), 'float32')
        self.convbn2d_98.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_98.weight', (128,1,5,5), 'float32')
        self.convbn2d_99.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_99.bias', (128), 'float32')
        self.convbn2d_99.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_99.weight', (128,128,1,1), 'float32')
        self.convbn2d_100.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_100.bias', (128), 'float32')
        self.convbn2d_100.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_100.weight', (128,1,5,5), 'float32')
        self.convbn2d_101.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_101.bias', (128), 'float32')
        self.convbn2d_101.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_101.weight', (128,128,1,1), 'float32')
        self.head_gfl_cls_2.bias = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.2.bias', (112), 'float32')
        self.head_gfl_cls_2.weight = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.2.weight', (112,128,1,1), 'float32')
        self.convbn2d_102.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_102.bias', (128), 'float32')
        self.convbn2d_102.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_102.weight', (128,1,5,5), 'float32')
        self.convbn2d_103.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_103.bias', (128), 'float32')
        self.convbn2d_103.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_103.weight', (128,128,1,1), 'float32')
        self.convbn2d_104.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_104.bias', (128), 'float32')
        self.convbn2d_104.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_104.weight', (128,1,5,5), 'float32')
        self.convbn2d_105.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_105.bias', (128), 'float32')
        self.convbn2d_105.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_105.weight', (128,128,1,1), 'float32')
        self.head_gfl_cls_3.bias = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.3.bias', (112), 'float32')
        self.head_gfl_cls_3.weight = self.load_pnnx_bin_as_parameter(archive, 'head.gfl_cls.3.weight', (112,128,1,1), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.convbn2d_0(v_0)
        v_2 = self.backbone_conv1_2(v_1)
        v_3 = self.backbone_maxpool(v_2)
        v_4 = self.convbn2d_1(v_3)
        v_5 = self.convbn2d_2(v_4)
        v_6 = self.backbone_stage2_0_branch1_4(v_5)
        v_7 = self.convbn2d_3(v_3)
        v_8 = self.backbone_stage2_0_branch2_2(v_7)
        v_9 = self.convbn2d_4(v_8)
        v_10 = self.convbn2d_5(v_9)
        v_11 = self.backbone_stage2_0_branch2_7(v_10)
        v_12 = torch.cat((v_6, v_11), dim=1)
        v_13 = self.channelshuffle_0(v_12)
        v_14, v_15 = torch.chunk(input=v_13, chunks=2, dim=1)
        v_16 = self.convbn2d_6(v_15)
        v_17 = self.backbone_stage2_1_branch2_2(v_16)
        v_18 = self.convbn2d_7(v_17)
        v_19 = self.convbn2d_8(v_18)
        v_20 = self.backbone_stage2_1_branch2_7(v_19)
        v_21 = torch.cat((v_14, v_20), dim=1)
        v_22 = self.channelshuffle_1(v_21)
        v_23, v_24 = torch.chunk(input=v_22, chunks=2, dim=1)
        v_25 = self.convbn2d_9(v_24)
        v_26 = self.backbone_stage2_2_branch2_2(v_25)
        v_27 = self.convbn2d_10(v_26)
        v_28 = self.convbn2d_11(v_27)
        v_29 = self.backbone_stage2_2_branch2_7(v_28)
        v_30 = torch.cat((v_23, v_29), dim=1)
        v_31 = self.channelshuffle_2(v_30)
        v_32, v_33 = torch.chunk(input=v_31, chunks=2, dim=1)
        v_34 = self.convbn2d_12(v_33)
        v_35 = self.backbone_stage2_3_branch2_2(v_34)
        v_36 = self.convbn2d_13(v_35)
        v_37 = self.convbn2d_14(v_36)
        v_38 = self.backbone_stage2_3_branch2_7(v_37)
        v_39 = torch.cat((v_32, v_38), dim=1)
        v_40 = self.channelshuffle_3(v_39)
        v_41 = self.convbn2d_15(v_40)
        v_42 = self.convbn2d_16(v_41)
        v_43 = self.backbone_stage3_0_branch1_4(v_42)
        v_44 = self.convbn2d_17(v_40)
        v_45 = self.backbone_stage3_0_branch2_2(v_44)
        v_46 = self.convbn2d_18(v_45)
        v_47 = self.convbn2d_19(v_46)
        v_48 = self.backbone_stage3_0_branch2_7(v_47)
        v_49 = torch.cat((v_43, v_48), dim=1)
        v_50 = self.channelshuffle_4(v_49)
        v_51, v_52 = torch.chunk(input=v_50, chunks=2, dim=1)
        v_53 = self.convbn2d_20(v_52)
        v_54 = self.backbone_stage3_1_branch2_2(v_53)
        v_55 = self.convbn2d_21(v_54)
        v_56 = self.convbn2d_22(v_55)
        v_57 = self.backbone_stage3_1_branch2_7(v_56)
        v_58 = torch.cat((v_51, v_57), dim=1)
        v_59 = self.channelshuffle_5(v_58)
        v_60, v_61 = torch.chunk(input=v_59, chunks=2, dim=1)
        v_62 = self.convbn2d_23(v_61)
        v_63 = self.backbone_stage3_2_branch2_2(v_62)
        v_64 = self.convbn2d_24(v_63)
        v_65 = self.convbn2d_25(v_64)
        v_66 = self.backbone_stage3_2_branch2_7(v_65)
        v_67 = torch.cat((v_60, v_66), dim=1)
        v_68 = self.channelshuffle_6(v_67)
        v_69, v_70 = torch.chunk(input=v_68, chunks=2, dim=1)
        v_71 = self.convbn2d_26(v_70)
        v_72 = self.backbone_stage3_3_branch2_2(v_71)
        v_73 = self.convbn2d_27(v_72)
        v_74 = self.convbn2d_28(v_73)
        v_75 = self.backbone_stage3_3_branch2_7(v_74)
        v_76 = torch.cat((v_69, v_75), dim=1)
        v_77 = self.channelshuffle_7(v_76)
        v_78, v_79 = torch.chunk(input=v_77, chunks=2, dim=1)
        v_80 = self.convbn2d_29(v_79)
        v_81 = self.backbone_stage3_4_branch2_2(v_80)
        v_82 = self.convbn2d_30(v_81)
        v_83 = self.convbn2d_31(v_82)
        v_84 = self.backbone_stage3_4_branch2_7(v_83)
        v_85 = torch.cat((v_78, v_84), dim=1)
        v_86 = self.channelshuffle_8(v_85)
        v_87, v_88 = torch.chunk(input=v_86, chunks=2, dim=1)
        v_89 = self.convbn2d_32(v_88)
        v_90 = self.backbone_stage3_5_branch2_2(v_89)
        v_91 = self.convbn2d_33(v_90)
        v_92 = self.convbn2d_34(v_91)
        v_93 = self.backbone_stage3_5_branch2_7(v_92)
        v_94 = torch.cat((v_87, v_93), dim=1)
        v_95 = self.channelshuffle_9(v_94)
        v_96, v_97 = torch.chunk(input=v_95, chunks=2, dim=1)
        v_98 = self.convbn2d_35(v_97)
        v_99 = self.backbone_stage3_6_branch2_2(v_98)
        v_100 = self.convbn2d_36(v_99)
        v_101 = self.convbn2d_37(v_100)
        v_102 = self.backbone_stage3_6_branch2_7(v_101)
        v_103 = torch.cat((v_96, v_102), dim=1)
        v_104 = self.channelshuffle_10(v_103)
        v_105, v_106 = torch.chunk(input=v_104, chunks=2, dim=1)
        v_107 = self.convbn2d_38(v_106)
        v_108 = self.backbone_stage3_7_branch2_2(v_107)
        v_109 = self.convbn2d_39(v_108)
        v_110 = self.convbn2d_40(v_109)
        v_111 = self.backbone_stage3_7_branch2_7(v_110)
        v_112 = torch.cat((v_105, v_111), dim=1)
        v_113 = self.channelshuffle_11(v_112)
        v_114 = self.convbn2d_41(v_113)
        v_115 = self.convbn2d_42(v_114)
        v_116 = self.backbone_stage4_0_branch1_4(v_115)
        v_117 = self.convbn2d_43(v_113)
        v_118 = self.backbone_stage4_0_branch2_2(v_117)
        v_119 = self.convbn2d_44(v_118)
        v_120 = self.convbn2d_45(v_119)
        v_121 = self.backbone_stage4_0_branch2_7(v_120)
        v_122 = torch.cat((v_116, v_121), dim=1)
        v_123 = self.channelshuffle_12(v_122)
        v_124, v_125 = torch.chunk(input=v_123, chunks=2, dim=1)
        v_126 = self.convbn2d_46(v_125)
        v_127 = self.backbone_stage4_1_branch2_2(v_126)
        v_128 = self.convbn2d_47(v_127)
        v_129 = self.convbn2d_48(v_128)
        v_130 = self.backbone_stage4_1_branch2_7(v_129)
        v_131 = torch.cat((v_124, v_130), dim=1)
        v_132 = self.channelshuffle_13(v_131)
        v_133, v_134 = torch.chunk(input=v_132, chunks=2, dim=1)
        v_135 = self.convbn2d_49(v_134)
        v_136 = self.backbone_stage4_2_branch2_2(v_135)
        v_137 = self.convbn2d_50(v_136)
        v_138 = self.convbn2d_51(v_137)
        v_139 = self.backbone_stage4_2_branch2_7(v_138)
        v_140 = torch.cat((v_133, v_139), dim=1)
        v_141 = self.channelshuffle_14(v_140)
        v_142, v_143 = torch.chunk(input=v_141, chunks=2, dim=1)
        v_144 = self.convbn2d_52(v_143)
        v_145 = self.backbone_stage4_3_branch2_2(v_144)
        v_146 = self.convbn2d_53(v_145)
        v_147 = self.convbn2d_54(v_146)
        v_148 = self.backbone_stage4_3_branch2_7(v_147)
        v_149 = torch.cat((v_142, v_148), dim=1)
        v_150 = self.channelshuffle_15(v_149)
        v_151 = self.convbn2d_55(v_40)
        v_152 = self.fpn_reduce_layers_0_act(v_151)
        v_153 = self.convbn2d_56(v_113)
        v_154 = self.fpn_reduce_layers_1_act(v_153)
        v_155 = self.convbn2d_57(v_150)
        v_156 = self.fpn_reduce_layers_2_act(v_155)
        v_157 = self.fpn_upsample(v_156)
        v_158 = torch.cat((v_157, v_154), dim=1)
        v_159 = self.convbn2d_58(v_158)
        v_160 = self.fpn_top_down_blocks_0_blocks_0_ghost1_primary_conv_2(v_159)
        v_161 = self.convbn2d_59(v_160)
        v_162 = self.fpn_top_down_blocks_0_blocks_0_ghost1_cheap_operation_2(v_161)
        v_163 = torch.cat((v_160, v_162), dim=1)
        v_164 = self.convbn2d_60(v_163)
        v_165 = self.convbn2d_61(v_164)
        v_166 = torch.cat((v_164, v_165), dim=1)
        v_167 = self.convbn2d_62(v_158)
        v_168 = self.convbn2d_63(v_167)
        v_169 = (v_166 + v_168)
        v_170 = self.pnnx_unique_0(v_169)
        v_171 = torch.cat((v_170, v_152), dim=1)
        v_172 = self.convbn2d_64(v_171)
        v_173 = self.fpn_top_down_blocks_1_blocks_0_ghost1_primary_conv_2(v_172)
        v_174 = self.convbn2d_65(v_173)
        v_175 = self.fpn_top_down_blocks_1_blocks_0_ghost1_cheap_operation_2(v_174)
        v_176 = torch.cat((v_173, v_175), dim=1)
        v_177 = self.convbn2d_66(v_176)
        v_178 = self.convbn2d_67(v_177)
        v_179 = torch.cat((v_177, v_178), dim=1)
        v_180 = self.convbn2d_68(v_171)
        v_181 = self.convbn2d_69(v_180)
        v_182 = (v_179 + v_181)
        v_183 = self.convbn2d_70(v_182)
        v_184 = self.fpn_downsamples_0_act(v_183)
        v_185 = self.convbn2d_71(v_184)
        v_186 = self.pnnx_unique_1(v_185)
        v_187 = torch.cat((v_186, v_169), dim=1)
        v_188 = self.convbn2d_72(v_187)
        v_189 = self.fpn_bottom_up_blocks_0_blocks_0_ghost1_primary_conv_2(v_188)
        v_190 = self.convbn2d_73(v_189)
        v_191 = self.fpn_bottom_up_blocks_0_blocks_0_ghost1_cheap_operation_2(v_190)
        v_192 = torch.cat((v_189, v_191), dim=1)
        v_193 = self.convbn2d_74(v_192)
        v_194 = self.convbn2d_75(v_193)
        v_195 = torch.cat((v_193, v_194), dim=1)
        v_196 = self.convbn2d_76(v_187)
        v_197 = self.convbn2d_77(v_196)
        v_198 = (v_195 + v_197)
        v_199 = self.convbn2d_78(v_198)
        v_200 = self.fpn_downsamples_1_act(v_199)
        v_201 = self.convbn2d_79(v_200)
        v_202 = self.pnnx_unique_2(v_201)
        v_203 = torch.cat((v_202, v_156), dim=1)
        v_204 = self.convbn2d_80(v_203)
        v_205 = self.fpn_bottom_up_blocks_1_blocks_0_ghost1_primary_conv_2(v_204)
        v_206 = self.convbn2d_81(v_205)
        v_207 = self.fpn_bottom_up_blocks_1_blocks_0_ghost1_cheap_operation_2(v_206)
        v_208 = torch.cat((v_205, v_207), dim=1)
        v_209 = self.convbn2d_82(v_208)
        v_210 = self.convbn2d_83(v_209)
        v_211 = torch.cat((v_209, v_210), dim=1)
        v_212 = self.convbn2d_84(v_203)
        v_213 = self.convbn2d_85(v_212)
        v_214 = (v_211 + v_213)
        v_215 = self.convbn2d_86(v_156)
        v_216 = self.fpn_extra_lvl_in_conv_0_act(v_215)
        v_217 = self.convbn2d_87(v_216)
        v_218 = self.pnnx_unique_3(v_217)
        v_219 = self.convbn2d_88(v_214)
        v_220 = self.fpn_extra_lvl_out_conv_0_act(v_219)
        v_221 = self.convbn2d_89(v_220)
        v_222 = self.pnnx_unique_4(v_221)
        v_223 = (v_218 + v_222)
        v_224 = self.convbn2d_90(v_182)
        v_225 = self.head_cls_convs_0_0_act(v_224)
        v_226 = self.convbn2d_91(v_225)
        v_227 = self.pnnx_unique_5(v_226)
        v_228 = self.convbn2d_92(v_227)
        v_229 = self.head_cls_convs_0_1_act(v_228)
        v_230 = self.convbn2d_93(v_229)
        v_231 = self.pnnx_unique_6(v_230)
        v_232 = self.head_gfl_cls_0(v_231)
        v_233 = torch.flatten(input=v_232, end_dim=-1, start_dim=2)
        v_234 = self.convbn2d_94(v_198)
        v_235 = self.head_cls_convs_1_0_act(v_234)
        v_236 = self.convbn2d_95(v_235)
        v_237 = self.pnnx_unique_7(v_236)
        v_238 = self.convbn2d_96(v_237)
        v_239 = self.head_cls_convs_1_1_act(v_238)
        v_240 = self.convbn2d_97(v_239)
        v_241 = self.pnnx_unique_8(v_240)
        v_242 = self.head_gfl_cls_1(v_241)
        v_243 = torch.flatten(input=v_242, end_dim=-1, start_dim=2)
        v_244 = self.convbn2d_98(v_214)
        v_245 = self.head_cls_convs_2_0_act(v_244)
        v_246 = self.convbn2d_99(v_245)
        v_247 = self.pnnx_unique_9(v_246)
        v_248 = self.convbn2d_100(v_247)
        v_249 = self.head_cls_convs_2_1_act(v_248)
        v_250 = self.convbn2d_101(v_249)
        v_251 = self.pnnx_unique_10(v_250)
        v_252 = self.head_gfl_cls_2(v_251)
        v_253 = torch.flatten(input=v_252, end_dim=-1, start_dim=2)
        v_254 = self.convbn2d_102(v_223)
        v_255 = self.head_cls_convs_3_0_act(v_254)
        v_256 = self.convbn2d_103(v_255)
        v_257 = self.pnnx_unique_11(v_256)
        v_258 = self.convbn2d_104(v_257)
        v_259 = self.head_cls_convs_3_1_act(v_258)
        v_260 = self.convbn2d_105(v_259)
        v_261 = self.pnnx_unique_12(v_260)
        v_262 = self.head_gfl_cls_3(v_261)
        v_263 = torch.flatten(input=v_262, end_dim=-1, start_dim=2)
        v_264 = torch.cat((v_233, v_243, v_253, v_263), dim=2)
        v_265 = v_264.permute(dims=(0,2,1))
        return v_265


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
              'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']


def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 416, 416, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("/home/bing/code/checkpoints/nanodet/nanodet_plus_m_1.5x_416_torchscript_pnnx.py.pt")


def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 416, 416, dtype=torch.float)

    torch.onnx._export(net, v_0, 
                       "/home/bing/code/checkpoints/nanodet/nanodet_plus_m_1.5x_416_torchscript_pnnx.py.onnx", 
                       export_params=True, 
                       operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, 
                       opset_version=13, 
                       input_names=['in0'], 
                       output_names=['out0'])


def get_single_level_center_priors(
        batch_size, featmap_size, stride, dtype, device
):
    h, w = featmap_size
    x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
    y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
    y, x = torch.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    strides = x.new_full((x.shape[0],), stride)
    proiors = torch.stack([x, y, strides, strides], dim=-1)
    return proiors.unsqueeze(0).repeat(batch_size, 1, 1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    keep = nms(boxes_for_nms, scores, iou_threshold=0.6)
    boxes = boxes[keep]
    scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels
    
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def preprocessing(img):
    # cv2.resize or cv2.warpPerspective
    # img = cv2.resize(img, (416, 416))
    M = np.array([[416/810, 0, 0],
                  [0, 416/1080, 0], 
                  [0, 0, 1]])
    dst_shape = (416, 416)
    img = cv2.warpPerspective(img, M, dsize=tuple(dst_shape))

    img = img.astype(np.float32) / 255.0
    # img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # rgb 
    img = (img - np.array([0.406, 0.456, 0.485])) / np.array([0.225, 0.224, 0.229])  # bgr 
    
    img = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float)
    img = torch.stack([img], dim=0).contiguous()
    return img


def postprocessing(preds):
    num_classes = 80
    reg_max = 7

    cls_scores, bbox_preds = preds.split(
        [num_classes, 4 * (reg_max + 1)], dim=-1
    )

    b = cls_scores.shape[0]
    input_height, input_width = 416, 416
    input_shape = (input_height, input_width)

    strides = [8, 16, 32, 64]

    featmap_sizes = [
        (math.ceil(input_height / stride), math.ceil(input_width) / stride)
        for stride in strides
    ]

    # get grid cells of one image
    mlvl_center_priors = [
        get_single_level_center_priors(
            b,
            featmap_sizes[i],
            stride,
            dtype=torch.float32, 
            device=cls_scores.device
        )
        for i, stride in enumerate(strides)
    ]

    #  
    center_priors = torch.cat(mlvl_center_priors, dim=1)

    # 
    # This approach, known as "Distribution Focal Loss" or "DFL", represents bounding 
    # box coordinates as discrete probability distributions. The softmax operation 
    # creates these distributions, and the linear projection converts them back to 
    # continuous coordinate values. This method can potentially improve the accuracy 
    # and stability of bounding box predictions in object detection tasks
    #
    shape = bbox_preds.size()
    project = torch.linspace(0, reg_max, reg_max + 1)
    bbox_preds = F.softmax(bbox_preds.reshape(*shape[:-1], 4, reg_max + 1), dim=-1)
    bbox_preds = F.linear(bbox_preds, project.type_as(bbox_preds)).reshape(*shape[:-1], 4)

    dis_preds = bbox_preds * center_priors[..., 2, None]
    bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
    scores = cls_scores.sigmoid()

    # batch nms 
    result_list = []
    for i in range(b):
        # add a dummy background class at the end of all labels
        # same with mmdetection2.0
        score, bbox = scores[i], bboxes[i]
        padding = score.new_zeros(score.shape[0], 1)
        score = torch.cat([score, padding], dim=1)
        results = multiclass_nms(
            bbox,
            score,
            score_thr=0.05,
            nms_cfg=dict(type="nms", iou_threshold=0.6),
            max_num=100,
        )
        result_list.append(results)

    ##################
    # rescale to original image size
    result = result_list[0]
    warp_matrix = np.array([[416/810, 0, 0], 
                            [0, 416/1080, 0], 
                            [0, 0, 1]])
    img_width, img_height = 810, 1080


    det_result = {}
    det_bboxes, det_labels = result
    det_bboxes = det_bboxes.detach().cpu().numpy()
    det_bboxes[:, :4] = warp_boxes(
        det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
    )

    # output format 
    classes = det_labels.detach().cpu().numpy()
    for i in range(num_classes):
        inds = classes == i
        det_result[i] = np.concatenate(
            [
                det_bboxes[inds, :4].astype(np.float32),
                det_bboxes[inds, 4:5].astype(np.float32),
            ],
            axis=1,
        ).tolist()

    return det_result


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img


def test_inference_ncnn():
    # fake input     
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 416, 416, dtype=torch.float)

    # real input 
    # preprocessing 
    raw_img = cv2.imread('/home/bing/code/open-source/rpi-deploy/data/bus.jpg')
    print(f'shape of raw image: {raw_img.shape}')
    print('print some values of raw image')
    print(raw_img[0, 0, :])
   
    in0 = preprocessing(raw_img)

    print(f'input shape: {in0.shape}')
    print('print some values of input tensor')
    print(in0[0, :, 0, 0])

    out = []
    with ncnn.Net() as net:
        net.load_param("/home/bing/code/checkpoints/nanodet/nanodet_plus_m_1.5x_416_torchscript.ncnn.param")
        net.load_model("/home/bing/code/checkpoints/nanodet/nanodet_plus_m_1.5x_416_torchscript.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))
    
    print(out[0][0, 0, -10:])
    # out.shape: 1*112*3598
    output = out[0].transpose(1, 2).contiguous()
    output = postprocessing(output)

    # visualization 
    result = overlay_bbox_cv(raw_img, 
                            output, 
                            class_names, 
                            score_thresh=0.35)

    cv2.imwrite("/home/bing/code/open-source/rpi-deploy/objection_detection/python/result_ncnn.jpg", result)
    
    return output


def test_inference_pnnx():
    net = Model()
    net.eval()

    # fake input 
    torch.manual_seed(0)    
    v_0 = torch.rand(1, 3, 416, 416, dtype=torch.float)

    # preprocessing 
    raw_img = cv2.imread('/home/bing/code/open-source/rpi-deploy/data/bus.jpg')
    print(f'shape of raw image: {raw_img.shape}')
    print('print some values of raw image')
    print(raw_img[0, :, 0, 0])

    v_0 = preprocessing(raw_img)

    # shape: 1*3598*112
    # 4 scales of feat map: 52*52, 26*26, 13*13, 7*7
    # 3589 = (416/8)^2 + (416/16)^2 + (416/32)^2 + (416/64)^2
    # 112 = 80 + 4 * (7+1)
    output = net(v_0)  
    output = postprocessing(output)

    # visualization 
    result = overlay_bbox_cv(raw_img, 
                            output, 
                            class_names, 
                            score_thresh=0.35)
    cv2.imwrite("/home/bing/code/open-source/rpi-deploy/objection_detection/python/result_pnnx.jpg", result)
    return output


if __name__ == "__main__":
    # result = test_inference_pnnx()
    result = test_inference_ncnn()
    # print(result)
