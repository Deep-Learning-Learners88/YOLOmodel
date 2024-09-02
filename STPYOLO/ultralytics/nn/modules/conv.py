# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math
from torch import nn, Tensor
from typing import Any, Callable
import math
import numpy as np
import torch
import torch.nn as nn
from torch import nn, Tensor
#from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
#from .transformer import TransformerBlock
#from ultralytics.utils.torch_utils import fuse_conv_and_bn
from typing import Optional
import torch.nn.functional as F

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "startConv","MOC2f","ColorAttentionModule",
    "OptimizedColorMoCAttention","MoCAttention","PPA","attention_model","Zoom_cat","Add","ScalSeq",
)


class Zoom_cat(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        #self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        # print(f"l shape: {l.shape}")
        # print(f"m shape: {m.shape}")
        # print(f"s shape: {s.shape}")

        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        #l = self.conv_l_post_down(l)
        # m = self.conv_m(m)
        # s = self.conv_s_pre_up(s)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        # s = self.conv_s_post_up(s)
        lms = torch.cat([l, m, s], dim=1)
        return lms
# class ScalSeq(nn.Module):
#     def __init__(self, channel):
#         super(ScalSeq, self).__init__()
#         self.conv1 =  Conv(128, channel,1)
#         self.conv2 =  Conv(256, channel,1)
#         self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
#         self.bn = nn.BatchNorm3d(channel)
#         self.act = nn.LeakyReLU(0.1)
#         self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))
#
#     def forward(self, x):
#         p3, p4, p5 = x[0],x[1],x[2]
#         print(f"l shape: {x[0].shape}")
#         print(f"m shape: {x[1].shape}")
#         print(f"s shape: {x[2].shape}")
#         p4_2 = self.conv1(p4)
#         p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
#         p5_2 = self.conv2(p5)
#         p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
#         p3_3d = torch.unsqueeze(p3, -3)
#         p4_3d = torch.unsqueeze(p4_2, -3)
#         p5_3d = torch.unsqueeze(p5_2, -3)
#         combine = torch.cat([p3_3d,p4_3d,p5_3d],dim = 2)
#         conv_3d = self.conv3d(combine)
#         bn = self.bn(conv_3d)
#         act = self.act(bn)
#         x = self.pool_3d(act)
#         x = torch.squeeze(x, 2)
#         return x
class ScalSeq(nn.Module):
    def __init__(self, channel):
        super(ScalSeq, self).__init__()
        # 调整 p3, p4, p5 的通道数为相同的 channel
        self.conv1 = Conv(64, channel, 1)  # p4
        self.conv2 = Conv(128, channel, 1)  # p5
        self.conv3 = Conv(32, channel, 1)   # p3
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        # print(f"l shape: {x[0].shape}")
        # print(f"m shape: {x[1].shape}")
        # print(f"s shape: {x[2].shape}")

        # 调整通道数
        p3_2 = self.conv3(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3_2.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3_2.size()[2:], mode='nearest')

        # 扩展维度以匹配3D卷积输入
        p3_3d = torch.unsqueeze(p3_2, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)

        # 拼接
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)

        # 3D卷积和后续操作
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)

        return x



class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()

    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


# class attention_model(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, ch=256):
#         super().__init__()
#         self.channel_att = channel_att(ch)
#         self.local_att = local_att(ch)
#
#     def forward(self, x):
#         input1, input2 = x[0], x[1]
#         input1 = self.channel_att(input1)
#         x = input1 + input2
#         x = self.local_att(x)
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class attention_model(nn.Module):
    def __init__(self, ch=256):
        super(attention_model, self).__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)
        self.adjust_channels = nn.Conv2d(32, 128, kernel_size=1)  # 假设 input1 的通道数是 32

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, "输入必须是包含两个张量的列表或元组"

        input1, input2 = x[0], x[1]


        # 将 input1 调整到与 input2 相同的通道数
        input1 = self.adjust_channels(input1)

        # 调整 input1 的空间维度，使其与 input2 匹配
        input1 = F.interpolate(input1, size=(input2.shape[2], input2.shape[3]), mode='nearest')

        # 对 input1 应用通道注意力
        input1 = self.channel_att(input1)

        # 将 input1 和 input2 相加
        x = input1 + input2

        # 对相加后的结果应用局部注意力
        x = self.local_att(x)

        return x



class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x
class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x
class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type='bn',
                                activation=False)
         self.c1 = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c2 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c3 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.bn1 = nn.BatchNorm2d(filters)
         self.drop = nn.Dropout2d(0.1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


# class OptimizedColorMoCAttention(nn.Module):
#     def __init__(self, channels, num_bins=32):
#         """
#         优化后的颜色注意力模块，采用新的主导颜色计算方法
#         :param channels: 输入和输出的通道数
#         :param num_bins: 颜色直方图的 bin 数，默认为 32，减少计算复杂度
#         """
#         super(OptimizedColorMoCAttention, self).__init__()
#         self.num_bins = num_bins
#         self.channels = channels
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#
#         # 将图像数据展平以进行像素值量化
#         reshaped_x = x.view(batch_size, channels, -1)
#
#         # 将像素值映射到指定的 bin 中
#         min_val, max_val = 0, 255
#         bin_size = (max_val - min_val) // self.num_bins
#         pixel_bins = (reshaped_x.float() // bin_size).long()  # 将像素值量化到 bins 中
#
#         # 初始化 bin 计数器
#         bin_counts = torch.zeros(batch_size, channels, self.num_bins, device=x.device)
#
#         # 对每个 bin 中的像素值进行计数
#         for i in range(self.num_bins):
#             bin_counts[:, :, i] = (pixel_bins == i).sum(dim=-1)  # 统计每个 bin 中的像素数
#
#         # 找到每个通道中最多的颜色 bin（主导颜色）
#         dominant_color_indices = bin_counts.argmax(dim=-1)
#
#         # 生成反注意力图
#         attention_maps = torch.ones_like(x)
#         for i in range(batch_size):
#             for c in range(channels):
#                 # 将每个像素值映射到对应的 bin
#                 pixel_bins = (x[i, c].float() // bin_size).long()
#                 # 生成反注意力图，忽略主导颜色
#                 attention_map = (pixel_bins != dominant_color_indices[i, c]).float()
#                 attention_maps[i, c] = attention_map
#
#         # 将反注意力图直接应用于输入特征
#         return x * attention_maps


class OptimizedColorMoCAttention(nn.Module):
    def __init__(
            self,
            InChannels: int,
            HidChannels: int = None,
            SqueezeFactor: int = 4,
            PoolRes: list = [1],
            Act: Callable[..., nn.Module] = nn.ReLU,
            ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
            MoCOrder: bool = True,
            num_bins: int = 32,
            **kwargs: Any,
    ) -> None:
        super().__init__()

        if HidChannels is None:
            HidChannels = max(self.makeDivisible(InChannels // SqueezeFactor, 8), 32)

        # 设置单个池化层，减少复杂性
        self.pooling_layer = nn.AdaptiveAvgPool2d(PoolRes[0])

        self.MoCOrder = MoCOrder
        self.num_bins = num_bins
        self.channels = InChannels

    def makeDivisible(self, v: int, divisor: int, min_value: int = None) -> int:
        """确保通道数可以被 divisor 整除"""
        if min_value is None:
            min_value = divisor
        return max(min_value, int(v + divisor / 2) // divisor * divisor)

    def shuffleTensor(self, x: Tensor) -> Tensor:
        """对张量进行随机打乱，尽量减少推理时的操作"""
        if not self.training:  # 推理时跳过打乱
            return x
        indices = torch.randperm(x.shape[0])
        return x[indices]

    def monteCarloSample(self, x: Tensor) -> Tensor:
        """执行简化的蒙特卡洛采样生成注意力图"""
        if self.training and self.MoCOrder:
            x = self.shuffleTensor(x)

        # 使用固定的池化层，去掉多余的均值操作
        return self.pooling_layer(x)

    def colorAttention(self, x: Tensor) -> Tensor:
        """计算颜色反注意力（最小化不必要的计算量）"""
        batch_size, channels, height, width = x.shape
        reshaped_x = x.view(batch_size * channels, -1).to(torch.float32).to(x.device)

        # 直接并行计算所有直方图，减少循环层次
        histograms = torch.histc(reshaped_x, bins=self.num_bins, min=0, max=255).view(batch_size, channels, self.num_bins)
        dominant_color_indices = histograms.argmax(dim=-1)

        # 简化生成反注意力图的过程，减少不必要的重复操作
        bin_size = 256 // self.num_bins
        pixel_bins = (x // bin_size).long()
        attention_maps = (pixel_bins != dominant_color_indices.unsqueeze(-1).unsqueeze(-1)).float()

        return attention_maps

    def forward(self, x: Tensor) -> Tensor:
        # 生成颜色反注意力图
        color_attention = self.colorAttention(x)

        # 生成蒙特卡洛注意力图
        monte_carlo_attention = self.monteCarloSample(x)

        # 将颜色反注意力图与蒙特卡洛注意力图相结合
        combined_attention = color_attention * monte_carlo_attention

        return x * combined_attention

# class ColorAttentionModule(nn.Module):
#     def __init__(self, channels, num_bins=32):
#         """
#         完善后的颜色注意力模块
#         :param channels: 输入和输出的通道数
#         :param num_bins: 颜色直方图的 bin 数，默认为 32，减少计算复杂度
#         """
#         super(ColorAttentionModule, self).__init__()
#         self.num_bins = num_bins
#         self.channels = channels
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#
#         # 将图像数据展平以计算直方图
#         reshaped_x = x.view(batch_size * channels, -1)
#
#         # 计算 bin_size 并量化图像数据
#         bin_size = 256 // self.num_bins
#         quantized_x = (reshaped_x // bin_size).long()
#
#         # 使用 clamp 限制 quantized_x 的值在 [0, num_bins-1] 之间
#         quantized_x = torch.clamp(quantized_x, 0, self.num_bins - 1)
#
#         # 使用 torch.bincount 计算每个通道的颜色直方图，并行计算，减少循环
#         histograms = torch.zeros(batch_size * channels, self.num_bins, device=x.device)
#         histograms.scatter_add_(1, quantized_x, torch.ones_like(quantized_x, dtype=torch.float32))
#
#         # 将 histograms reshape 为 (batch_size, channels, num_bins)
#         histograms = histograms.view(batch_size, channels, self.num_bins)
#
#         # 找到每个通道中最多的颜色（主导颜色）
#         dominant_color_indices = histograms.argmax(dim=-1)
#
#         # 生成反注意力图，利用广播机制处理
#         pixel_bins = (x.float() // bin_size).long()
#         dominant_color_indices = dominant_color_indices.unsqueeze(-1).unsqueeze(-1)
#         attention_maps = (pixel_bins != dominant_color_indices).float()
#
#         # 将反注意力图直接应用于输入特征
#         return x * attention_maps


# class ColorAttentionModule(nn.Module):
#     def __init__(self, channels, num_bins=32):
#         """
#         优化后的颜色注意力模块
#         :param channels: 输入和输出的通道数
#         :param num_bins: 颜色直方图的 bin 数，默认为 32，减少计算复杂度
#         """
#         super(ColorAttentionModule, self).__init__()
#         self.num_bins = num_bins
#         self.channels = channels
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#
#         # 将图像数据展平以计算直方图
#         reshaped_x = x.view(batch_size * channels, -1)
#
#         # 计算每个通道的颜色直方图
#         histograms = torch.stack([
#             torch.histc(reshaped_x[i], bins=self.num_bins, min=0, max=255)
#             for i in range(batch_size * channels)
#         ]).view(batch_size, channels, self.num_bins)
#
#         # 找到每个通道中最多的颜色（主导颜色）
#         dominant_color_indices = histograms.argmax(dim=-1)
#
#         # 生成反注意力图
#         bin_size = 256 // self.num_bins
#         attention_maps = torch.ones_like(x)
#         for i in range(batch_size):
#             for c in range(channels):
#                 pixel_bins = (x[i, c].float() // bin_size).long()
#                 attention_map = (pixel_bins != dominant_color_indices[i, c]).float()
#                 attention_maps[i, c] = attention_map
#
#         # 将反注意力图直接应用于输入特征
#         return x * attention_maps
class ColorAttentionModule(nn.Module):
    def __init__(self, channels, num_bins=256):
        """
        初始化颜色注意力模块
        :param channels: 输入和输出的通道数
        :param num_bins: 颜色直方图的 bin 数，默认为 256
        """
        super(ColorAttentionModule, self).__init__()
        self.num_bins = num_bins
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 将图像数据展平以计算直方图
        reshaped_x = x.view(batch_size * channels, -1)

        # 计算颜色直方图
        histograms = []
        for i in range(batch_size * channels):
            hist = torch.histc(reshaped_x[i].float(), bins=self.num_bins, min=0, max=255)
            histograms.append(hist)
        histograms = torch.stack(histograms).view(batch_size, channels, self.num_bins)

        # 找到每个通道最多的颜色
        dominant_color_indices = histograms.argmax(dim=-1)

        # 生成注意力图
        bin_size = 256 // self.num_bins
        attention_maps = torch.ones_like(x)
        for i in range(batch_size):
            for c in range(channels):
                pixel_bins = (x[i, c].float() // bin_size).long()
                attention_map = (pixel_bins != dominant_color_indices[i, c]).float()
                attention_maps[i, c] = attention_map

        # 使用卷积和批归一化生成最终的注意力图
        attention_maps = self.conv1(attention_maps)
        attention_maps = self.bn1(attention_maps)
        attention_maps = self.relu(attention_maps)
        attention_maps = self.conv2(attention_maps)
        attention_maps = self.bn2(attention_maps)
        attention_maps = self.sigmoid(attention_maps)

        # 将注意力图应用于输入特征
        x = x * attention_maps

        return x

# class ColorAttentionModule(nn.Module):
#     def __init__(self, channels, num_bins=256):
#         """
#         初始化颜色注意力模块
#         :param channels: 输入和输出的通道数
#         :param num_bins: 颜色直方图的 bin 数，默认为 256
#         """
#         super(ColorAttentionModule, self).__init__()
#         self.num_bins = num_bins
#         self.channels = channels
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.conv2 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
#         self.bn2 = nn.BatchNorm2d(1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#
#         # 将图像数据展平以计算直方图
#         reshaped_x = x.view(batch_size * channels, -1)
#
#         # 计算颜色直方图
#         histograms = []
#         for i in range(batch_size * channels):
#             hist = torch.histc(reshaped_x[i].float(), bins=self.num_bins, min=0, max=255)
#             histograms.append(hist)
#         histograms = torch.stack(histograms).view(batch_size, channels, self.num_bins)
#
#         # 找到每个通道最多的颜色
#         dominant_color_indices = histograms.argmax(dim=-1)
#
#         # 生成注意力图
#         bin_size = 256 // self.num_bins
#         attention_maps = torch.ones_like(x)
#         for i in range(batch_size):
#             for c in range(channels):
#                 pixel_bins = (x[i, c].float() // bin_size).long()
#                 attention_map = (pixel_bins != dominant_color_indices[i, c]).float()
#                 attention_maps[i, c] = attention_map
#
#         # 使用卷积和批归一化生成最终的注意力图
#         attention_maps = self.conv1(attention_maps)
#         attention_maps = self.bn1(attention_maps)
#         attention_maps = self.relu(attention_maps)
#         attention_maps = self.conv2(attention_maps)
#         attention_maps = self.bn2(attention_maps)
#         attention_maps = self.sigmoid(attention_maps)
#
#         # 将注意力图应用于输入特征
#         x = x * attention_maps
#
#        return x


# class OptimizedColorMoCAttention(nn.Module):
#     def __init__(
#             self,
#             InChannels: int,
#             HidChannels: int = None,
#             SqueezeFactor: int = 4,
#             PoolRes: list = [1],
#             Act: Callable[..., nn.Module] = nn.ReLU,
#             ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
#             MoCOrder: bool = True,
#             num_bins: int = 32,
#             **kwargs: Any,
#     ) -> None:
#         super().__init__()
#
#         if HidChannels is None:
#             HidChannels = max(self.makeDivisible(InChannels // SqueezeFactor, 8), 32)
#
#         # 设置单个池化层，减少复杂性
#         self.pooling_layer = nn.AdaptiveAvgPool2d(PoolRes[0])
#
#         self.MoCOrder = MoCOrder
#         self.num_bins = num_bins
#         self.channels = InChannels
#
#     def makeDivisible(self, v: int, divisor: int, min_value: int = None) -> int:
#         """确保通道数可以被 divisor 整除"""
#         if min_value is None:
#             min_value = divisor
#         new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#         return new_v
#
#     def shuffleTensor(self, x: Tensor) -> Tensor:
#         """对张量进行随机打乱"""
#         indices = torch.randperm(x.shape[0])
#         return x[indices], indices
#
#     def monteCarloSample(self, x: Tensor) -> Tensor:
#         """执行简化的蒙特卡洛采样生成注意力图"""
#         if self.training:
#             x1 = self.shuffleTensor(x)[0] if self.MoCOrder else x  # 打乱张量
#         else:
#             x1 = x
#
#         # 使用固定的池化层来减少随机性
#         AttnMap: Tensor = self.pooling_layer(x1)  # 使用单一池化层操作
#         if AttnMap.shape[-1] > 1:  # 如果注意力图有多个元素，压缩维度
#             AttnMap = AttnMap.mean(dim=-1, keepdim=True)  # 取均值代替随机选择
#
#         return AttnMap
#
#     def colorAttention(self, x: Tensor) -> Tensor:
#         """计算颜色反注意力（使用GPU并行加速）"""
#         batch_size, channels, height, width = x.shape
#
#         # 强制将输入转换为 float32 以确保 torch.histc 支持
#         reshaped_x = x.view(batch_size * channels, -1).to(torch.float32).to(x.device)
#
#         # 初始化 GPU 上的直方图张量
#         histograms = torch.zeros(batch_size * channels, self.num_bins, device=x.device)
#
#         # 并行计算每个通道的颜色直方图
#         histograms = torch.stack([
#             torch.histc(reshaped_x[i], bins=self.num_bins, min=0, max=255)
#             for i in range(batch_size * channels)
#         ]).view(batch_size, channels, self.num_bins)
#
#         # 找到每个通道中最多的颜色（主导颜色）
#         dominant_color_indices = histograms.argmax(dim=-1)
#
#         # 生成反注意力图
#         bin_size = 256 // self.num_bins
#         attention_maps = torch.ones_like(x, device=x.device)
#         for i in range(batch_size):
#             for c in range(channels):
#                 # 将像素值分配到相应的 bin 中
#                 pixel_bins = (x[i, c].float() // bin_size).long()
#                 # 生成反注意力图，忽略主导颜色
#                 attention_map = (pixel_bins != dominant_color_indices[i, c]).float()
#                 attention_maps[i, c] = attention_map
#
#         return attention_maps
#
#     def forward(self, x: Tensor) -> Tensor:
#         # 生成颜色反注意力图
#         color_attention = self.colorAttention(x)
#
#         # 生成蒙特卡洛注意力图
#         monte_carlo_attention = self.monteCarloSample(x)
#
#         # 将颜色反注意力图与蒙特卡洛注意力图相结合
#         combined_attention = color_attention * monte_carlo_attention
#
#         # 输出结合后的注意力图
#         return x * combined_attention

####yuanlai
# class OptimizedColorMoCAttention(nn.Module):
#     def __init__(
#             self,
#             InChannels: int,
#             HidChannels: int = None,
#             SqueezeFactor: int = 4,
#             PoolRes: list = [1],  # 固定为一个池化层分辨率来减少随机采样
#             Act: Callable[..., nn.Module] = nn.ReLU,
#             ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
#             MoCOrder: bool = True,
#             num_bins: int = 32,  # 降低 num_bins 来减少计算量
#             **kwargs: Any,
#     ) -> None:
#         super().__init__()
#
#         if HidChannels is None:
#             HidChannels = max(self.makeDivisible(InChannels // SqueezeFactor, 8), 32)
#
#         # 设置单个池化层，减少复杂性
#         self.pooling_layer = nn.AdaptiveAvgPool2d(PoolRes[0])
#
#         # SELayer用于处理蒙特卡洛注意力
#         self.SELayer = nn.Sequential(
#             nn.Conv2d(InChannels, HidChannels, kernel_size=1),
#             Act(),
#             nn.Conv2d(HidChannels, InChannels, kernel_size=1),
#             ScaleAct()
#         )
#
#         self.MoCOrder = MoCOrder
#         self.num_bins = num_bins
#         self.channels = InChannels
#
#     def makeDivisible(self, v: int, divisor: int, min_value: int = None) -> int:
#         """确保通道数可以被 divisor 整除"""
#         if min_value is None:
#             min_value = divisor
#         new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#         return new_v
#
#     def shuffleTensor(self, x: Tensor) -> Tensor:
#         """对张量进行随机打乱"""
#         indices = torch.randperm(x.shape[0])
#         return x[indices], indices
#
#     def monteCarloSample(self, x: Tensor) -> Tensor:
#         """执行简化的蒙特卡洛采样生成注意力图"""
#         if self.training:
#             x1 = self.shuffleTensor(x)[0] if self.MoCOrder else x  # 打乱张量
#         else:
#             x1 = x
#
#         # 使用固定的池化层来减少随机性
#         AttnMap: Tensor = self.pooling_layer(x1)  # 使用单一池化层操作
#         if AttnMap.shape[-1] > 1:  # 如果注意力图有多个元素，压缩维度
#             AttnMap = AttnMap.mean(dim=-1, keepdim=True)  # 取均值代替随机选择
#
#         return AttnMap
#
#     def colorAttention(self, x: Tensor) -> Tensor:
#         """计算颜色反注意力（使用GPU并行加速）"""
#         batch_size, channels, height, width = x.shape
#
#         # 强制将输入转换为 float32 以确保 torch.histc 支持
#         reshaped_x = x.view(batch_size * channels, -1).to(torch.float32).to(x.device)
#
#         # 初始化 GPU 上的直方图张量
#         histograms = torch.zeros(batch_size * channels, self.num_bins, device=x.device)
#
#         # 并行计算每个通道的颜色直方图
#         histograms = torch.stack([
#             torch.histc(reshaped_x[i], bins=self.num_bins, min=0, max=255)
#             for i in range(batch_size * channels)
#         ]).view(batch_size, channels, self.num_bins)
#
#         # 找到每个通道中最多的颜色（主导颜色）
#         dominant_color_indices = histograms.argmax(dim=-1)
#
#         # 生成反注意力图
#         bin_size = 256 // self.num_bins
#         attention_maps = torch.ones_like(x, device=x.device)
#         for i in range(batch_size):
#             for c in range(channels):
#                 # 将像素值分配到相应的 bin 中
#                 pixel_bins = (x[i, c].float() // bin_size).long()
#                 # 生成反注意力图，忽略主导颜色
#                 attention_map = (pixel_bins != dominant_color_indices[i, c]).float()
#                 attention_maps[i, c] = attention_map
#
#         return attention_maps
#
#     def forward(self, x: Tensor) -> Tensor:
#         # 生成颜色反注意力图
#         color_attention = self.colorAttention(x)
#
#         # 生成蒙特卡洛注意力图
#         monte_carlo_attention = self.monteCarloSample(x)
#
#         # 将颜色反注意力图与蒙特卡洛注意力图相结合
#         combined_attention = color_attention * monte_carlo_attention
#
#         # 应用SELayer处理结合后的注意力图
#         return x * self.SELayer(combined_attention)

###
#import torch


# class OptimizedColorMoCAttention(nn.Module):
#     def __init__(
#             self,
#             InChannels: int,
#             HidChannels: int = None,
#             SqueezeFactor: int = 4,
#             PoolRes: list = [1],
#             Act: Callable[..., nn.Module] = nn.ReLU,
#             ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
#             MoCOrder: bool = True,
#             num_bins: int = 32,  # 降低 num_bins 来减少计算量
#             block_size: int = 32,  # 每个块的大小，用于分块处理
#             **kwargs: Any,
#     ) -> None:
#         super().__init__()
#
#         if HidChannels is None:
#             HidChannels = max(self.makeDivisible(InChannels // SqueezeFactor, 8), 32)
#
#         # 设置单个池化层
#         self.pooling_layer = nn.AdaptiveAvgPool2d(PoolRes[0])
#
#         # SELayer用于处理蒙特卡洛注意力
#         self.SELayer = nn.Sequential(
#             nn.Conv2d(InChannels, HidChannels, kernel_size=1),
#             Act(),
#             nn.Conv2d(HidChannels, InChannels, kernel_size=1),
#             ScaleAct()
#         )
#
#         self.MoCOrder = MoCOrder
#         self.num_bins = num_bins
#         self.channels = InChannels
#         self.block_size = block_size  # 定义每个块的大小
#
#     def makeDivisible(self, v: int, divisor: int, min_value: int = None) -> int:
#         """确保通道数可以被 divisor 整除"""
#         if min_value is None:
#             min_value = divisor
#         new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#         return new_v
#
#     def shuffleTensor(self, x: Tensor) -> Tensor:
#         """对张量进行随机打乱"""
#         indices = torch.randperm(x.shape[0])
#         return x[indices], indices
#
#     def monteCarloSample(self, x: Tensor) -> Tensor:
#         """执行简化的蒙特卡洛采样生成注意力图"""
#         if self.training:
#             x1 = self.shuffleTensor(x)[0] if self.MoCOrder else x  # 打乱张量
#         else:
#             x1 = x
#
#         # 使用固定的池化层来减少随机性
#         AttnMap: Tensor = self.pooling_layer(x1)  # 使用单一池化层操作
#         if AttnMap.shape[-1] > 1:  # 如果注意力图有多个元素，压缩维度
#             AttnMap = AttnMap.mean(dim=-1, keepdim=True)  # 取均值代替随机选择
#
#         return AttnMap
#
#     def colorAttention(self, x: Tensor) -> Tensor:
#         """基于块的颜色反注意力（使用GPU并行加速）"""
#         batch_size, channels, height, width = x.shape
#
#         # 计算每个维度上的块数
#         num_blocks_h = height // self.block_size
#         num_blocks_w = width // self.block_size
#
#         # 初始化 GPU 上的反注意力图张量
#         attention_maps = torch.ones_like(x, device=x.device)
#
#         # 遍历每个块
#         for i in range(num_blocks_h):
#             for j in range(num_blocks_w):
#                 # 提取当前块
#                 block_x = x[:, :, i * self.block_size:(i + 1) * self.block_size, j * self.block_size:(j + 1) * self.block_size]
#                 reshaped_block_x = block_x.view(batch_size * channels, -1).to(torch.float32).to(x.device)
#
#                 # 初始化 GPU 上的直方图张量
#                 histograms = torch.zeros(batch_size * channels, self.num_bins, device=x.device)
#
#                 # 并行计算每个通道的颜色直方图
#                 histograms = torch.stack([
#                     torch.histc(reshaped_block_x[k], bins=self.num_bins, min=0, max=255)
#                     for k in range(batch_size * channels)
#                 ]).view(batch_size, channels, self.num_bins)
#
#                 # 找到每个通道中最多的颜色（主导颜色）
#                 dominant_color_indices = histograms.argmax(dim=-1)
#
#                 # 生成反注意力图
#                 bin_size = 256 // self.num_bins
#                 for b in range(batch_size):
#                     for c in range(channels):
#                         # 将像素值分配到相应的 bin 中
#                         pixel_bins = (block_x[b, c].float() // bin_size).long()
#                         # 生成反注意力图，忽略主导颜色
#                         attention_map_block = (pixel_bins != dominant_color_indices[b * channels + c]).float()
#                         attention_maps[b, c, i * self.block_size:(i + 1) * self.block_size, j * self.block_size:(j + 1) * self.block_size] = attention_map_block
#
#         return attention_maps
#
#     def forward(self, x: Tensor) -> Tensor:
#         # 生成基于块的颜色反注意力图
#         color_attention = self.colorAttention(x)
#
#         # 生成蒙特卡洛注意力图
#         monte_carlo_attention = self.monteCarloSample(x)
#
#         # 将颜色反注意力图与蒙特卡洛注意力图相结合
#         combined_attention = color_attention * monte_carlo_attention
#
#         # 应用SELayer处理结合后的注意力图
#         return x * self.SELayer(combined_attention)


#

###3
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

#mo注意力

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0





def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.Py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



NormLayerTuple = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.GroupNorm,
    nn.BatchNorm3d,
)

def initWeight(Module):
    # init conv, norm , and linear layers
    ## empty module
    if Module is None:
        return
    ## conv layer
    elif isinstance(Module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(Module.bias, -bound, bound)
    ## norm layer
    elif isinstance(Module, NormLayerTuple):
        if Module.weight is not None:
            nn.init.ones_(Module.weight)
        if Module.bias is not None:
            nn.init.zeros_(Module.bias)
    ## linear layer
    elif isinstance(Module, nn.Linear):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(Module.bias, -bound, bound)
    elif isinstance(Module, (nn.Sequential, nn.ModuleList)):
        for m in Module:
            initWeight(m)
    elif list(Module.children()):
        for m in Module.children():
            initWeight(m)



def pair(Val):
    return Val if isinstance(Val, (tuple, list)) else (Val, Val)

class BaseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            groups: Optional[int] = 1,
            bias: Optional[bool] = None,
            BNorm: bool = False,
            # norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
            ActLayer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            Momentum: Optional[float] = 0.1,
            **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)

        if bias is None:
            bias = not BNorm

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.Conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)

        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()

        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer

        self.apply(initWeight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

    def profileModule(self, Input: Tensor):
        if Input.dim() != 4:
            print('Conv2d requires 4-dimensional Input (BxCxHxW). Provided Input has shape: {}'.format(Input.size()))

        BatchSize, in_channels, in_h, in_w = Input.size()
        assert in_channels == self.in_channels, '{}!={}'.format(in_channels, self.in_channels)

        k_h, k_w = pair(self.kernel_size)
        stride_h, stride_w = pair(self.stride)
        pad_h, pad_w = pair(self.padding)
        groups = self.groups

        out_h = (in_h - k_h + 2 * pad_h) // stride_h + 1
        out_w = (in_w - k_w + 2 * pad_w) // stride_w + 1

        # compute MACs
        MACs = (k_h * k_w) * (in_channels * self.out_channels) * (out_h * out_w) * 1.0
        MACs /= groups

        if self.bias:
            MACs += self.out_channels * out_h * out_w

        # compute parameters
        Params = sum([p.numel() for p in self.parameters()])

        Output = torch.zeros(size=(BatchSize, self.out_channels, out_h, out_w), dtype=Input.dtype, device=Input.device)
        # print(MACs)
        return Output, Params, MACs


def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)


import numpy as np
class MoCAttention(nn.Module):
    # Monte carlo attention
    def __init__(
            self,
            InChannels: int,
            HidChannels: int = None,
            SqueezeFactor: int = 4,
            PoolRes: list = [1, 2],
            Act: Callable[..., nn.Module] = nn.ReLU,
            ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
            MoCOrder: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)

        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)

        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )

        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder

    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None]  # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)

        return AttnMap

    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        return x * self.SELayer(AttnMap)

def callMethod(self, ElementName):
    return getattr(self, ElementName)
def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    # shuffle multiple tensors with the same indexs
    # all tensors must have the same shape
    if isinstance(Feature, Tensor):
        Feature = [Feature]

    Indexs = None
    Output = []
    for f in Feature:
        # not in-place operation, should update output
        B, C, H, W = f.shape
        if Mode == 1:
            # fully shuffle
            f = f.flatten(2)
            if Indexs is None:
                Indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, Indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        else:
            # shuflle along y and then x axis
            if Indexs is None:
                Indexs = [torch.randperm(H, device=f.device),
                          torch.randperm(W, device=f.device)]
            f = f[:, :, Indexs[0].to(f.device)]
            f = f[:, :, :, Indexs[1].to(f.device)]
        Output.append(f)
    return Output



class MOC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with MoCAttention with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        # 替换 Bottleneck 为 MoCAttention
        self.m = nn.ModuleList(MoCAttention(self.c, HidChannels=self.c, MoCOrder=True) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)  # 使用 MoCAttention
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)  # 使用 MoCAttention
        return self.cv2(torch.cat(y, 1))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class startConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # 新增 startBlock 层
        self.start_block = startBlock(c2, mlp_ratio=3)

    def forward(self, x):
        """Apply convolution, batch normalization, and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))

        # 调用 startBlock 层的操作
        x = self.start_block(x)

        return x
from timm.models.layers import DropPath, trunc_normal_

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class startBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        # print(x[0].shape)
        # print(x[1].shape)

        return torch.cat(x, self.d)
