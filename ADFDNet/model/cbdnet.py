
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class filter(nn.Module):
    def __init__(self):
        super(filter, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return filter_init(x)

# 定义高斯滤波器
def gaussian_filter(kernel_size, sigma):
    coords = torch.arange(kernel_size).float()
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.view(1, 1, kernel_size, 1)

def filter_init(image, kernel_size=5, sigma=1):
    kernel = gaussian_filter(kernel_size, sigma)
    kernel = kernel.repeat(1, 1, 1, kernel_size)
    kernel = kernel.to(image.device)
    blurred = F.conv2d(image, kernel, padding=kernel_size // 2)
    high_pass = image - blurred

    return  high_pass,  blurred

class fuse(nn.Module):
    # def __init__(self, alpha_h = 0.5, alpha_l = 0.5):
    def __init__(self, alpha=0.5):
        super(fuse, self).__init__()
        # self.requires_grad = False
        # self.alpha = alpha
        features = 64
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # self.alpha_h = nn.Parameter(torch.tensor(alpha_h))
        # self.alpha_l = nn.Parameter(torch.tensor(alpha_l))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.outconv = nn.Conv2d(features, 1, 1)



    def forward(self, x_h, x_l):

        # alpha_h = torch.sigmoid(self.alpha_h)
        # alpha_l = torch.sigmoid(self.alpha_l)
        alpha_h = torch.sigmoid(self.alpha)

        # fused_image = self.alpha_h * x_h + self.alpha_l * x_l
        fused_image = self.alpha * x_h + (1 - alpha )  * x_l
        fused_image = self.conv1(fused_image)
        fused_image = self.conv2(fused_image)
        fused_image = self.outconv(fused_image)

        return fused_image, alpha_h, alpha_l


class AdaptiveFuse(nn.Module):
    def __init__(self):
        super(AdaptiveFuse, self).__init__()
        features = 64

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)  # 用于生成自适应权重
        self.conv2 = nn.Conv2d(features, 1, kernel_size=3, padding=1)  # 后续卷积操作

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

    def forward(self, x_h, x_l):
        # 拼接高频和低频特征
        combined = torch.cat([x_h, x_l], dim=1)

        # 生成自适应权重
        attention_weights = torch.sigmoid(self.conv1(combined))  # 生成一个0-1之间的权重图

        # 应用自适应权重进行融合
        fused_image = attention_weights * x_h + (1 - attention_weights) * x_l
        fused_image = self.conv_1(fused_image)
        fused_image = self.conv_2(fused_image)

        # 后续处理
        fused_image = self.conv2(fused_image)
        return fused_image

class noFuse(nn.Module):
    def __init__(self):
        super( noFuse, self).__init__()
        features = 64
        self.weights = 0.5

        self.conv2 = nn.Conv2d(features, 1, kernel_size=3, padding=1)  # 后续卷积操作

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

    def forward(self, x_h, x_l):

        # 融合
        fused_image = self.weights * x_h + (1 - self.weights) * x_l
        fused_image = self.conv_1(fused_image)
        fused_image = self.conv_2(fused_image)

        # 后续处理
        fused_image = self.conv2(fused_image)
        return fused_image

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class NET1(nn.Module):
    def __init__(self):
        super(NET1, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net1(x)


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fcn(x)


# 定义SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        batch, channel, _, _ = x.size()
        # 全局平均池化
        y = F.adaptive_avg_pool2d(x, 1).view(batch, channel)
        # 全连接层和激活函数
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch, channel, 1, 1)
        # 乘以原输入
        return x * y

class SB(nn.Module):
    def __init__(self):
        super(SB, self).__init__()

        channels = 2
        features = 96

        # Initial layer
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, groups=1, bias=False, dilation=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, groups=1, bias=False, dilation=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3,  padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, groups=1, bias=False, dilation=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        # self.conv1_12 = nn.Sequential(
        #     nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, groups=1, bias=False, dilation=2),
        #     nn.BatchNorm2d(features),
        #     nn.ReLU(inplace=True))
        self.SE = SEBlock(features)
        self.outc = outconv(features*2, 1)

    def forward(self, x):

        x1 = self.inc(x)

        x = self.conv1_1(x1)
        x = self.conv1_2(self.SE(x)) + x
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = self.conv1_5(self.SE(x)) + x
        x = self.conv1_6(x)
        x = self.conv1_7(x)
        x = self.conv1_8(x)
        x = self.conv1_9(self.SE(x)) + x

        x = self.conv1_10(x)
        x = self.conv1_11(x)

        # x_input = x[:, 1, :, :].unsqueeze(1)
        # out = x_input - self.outc(x)
        out = torch.cat([x1, x], 1)  # 通道拼接
        out = self.outc(out)

        return out

class noSB(nn.Module):
    def __init__(self):
        super(noSB, self).__init__()

        channels = 2
        features = 96

        # Initial layer
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3,  padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True))

        # self.conv1_12 = nn.Sequential(
        #     nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, groups=1, bias=False, dilation=2),
        #     nn.BatchNorm2d(features),
        #     nn.ReLU(inplace=True))
        self.SE = SEBlock(features)
        self.outc = outconv(features*2, 1)

    def forward(self, x):

        x1 = self.inc(x)

        x = self.conv1_1(x1)
        x = self.conv1_2(self.SE(x)) + x
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = self.conv1_5(self.SE(x)) + x
        x = self.conv1_6(x)
        x = self.conv1_7(x)
        x = self.conv1_8(x)
        x = self.conv1_9(self.SE(x)) + x

        x = self.conv1_10(x)
        x = self.conv1_11(x)

        # x_input = x[:, 1, :, :].unsqueeze(1)
        # out = x_input - self.outc(x)
        out = torch.cat([x1, x], 1)  # 通道拼接
        out = self.outc(out)

        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用深度卷积进行空间注意力
        # kernel_size为7通常效果较好，可以尝试不同的大小
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        # 使用最大池化和平均池化获取通道间的空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 拼接最大池化和平均池化的结果
        x = torch.cat([max_out, avg_out], dim=1)
        # 使用卷积处理拼接的特征，并通过sigmoid得到空间注意力
        x = self.conv(x)
        return self.sigmoid(x) * input

# class ResEnhanceNet(nn.Module):
#     def __init__(self):
#         super(ResEnhanceNet, self).__init__()
#
#         channels = 2
#         features = 96
#         kernel_size = 3
#
#         self.inc = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,  bias=False),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv2_1 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,  groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_2 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_3 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_4 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_5 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_6 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_7 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_8 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.conv2_9 = nn.Sequential(
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1, bias=False))
#
#         self.SA = SpatialAttention()
#         self.outc = outconv(channels, 1)
#
#     def forward(self, input):
#
#         x = self.inc(input)
#
#         res1 = self.conv2_1(x)
#         res1 += x
#
#         res2 = self.conv2_2(res1)
#         res2 += res1
#
#         res3 = self.conv2_3(res2)
#         res3 += res2
#
#         res4 = self.conv2_4(res3)
#         res4 += res3
#
#         res5 = self.conv2_5(res4)
#         res5 += res4
#
#         res6 = self.conv2_6(res5)
#         res6 += res5
#
#         res7 = self.conv2_7(res6)
#         res7 += res6
#
#         res8 = self.conv2_8(res7)
#         res8 += res7
#
#         res9 = self.conv2_9(res8)
#         res9 += res8
#
#         out = self.SA(res9)
#
#         input = input[:, 0, :, :].unsqueeze(1)
#         out = torch.cat([input, out], 1)
#
#         out = self.outc(out)
#
#         return out

class ResEnhanceNet(nn.Module):
    def __init__(self):
        super(ResEnhanceNet, self).__init__()

        channels = 2
        features = 80
        mult_features = 120
        kernel_size = 3

        # 初始卷积层
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 多尺度卷积层
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1, bias=False),  # 3×3卷积
            nn.Conv2d(features, features // 2, kernel_size=5, padding=2, bias=False),  # 5×5卷积
            nn.Conv2d(features, features // 2, kernel_size=7, padding=3, bias=False)   # 7×7卷积
        ])

        # 残差模块
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mult_features, mult_features, kernel_size=kernel_size, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mult_features, mult_features, kernel_size=kernel_size, padding=1, bias=False)
            ) for _ in range(6)
        ])

        # 空间注意力模块
        self.SA = SpatialAttention()

        # 输出卷积
        self.outc = outconv(200, 1)

    def forward(self, input):
        # 初始特征提取
        x1 = self.inc(input)

        # 多尺度特征提取
        multi_scale_features = [conv(x1) for conv in self.multi_scale]
        x = torch.cat(multi_scale_features, dim=1)  # 通道拼接

        # 空间注意力
        x = self.SA(x)


        # 残差模块
        for block in self.res_blocks:
            res = block(x)
            x = x + res

        # 融合输入和输出
        out = torch.cat([x1, x], 1)  # 通道拼接

        # 最终输出
        out = self.outc(out)

        return out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(2, 32),
            single_conv(32, 32)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(32, 64),
            # single_conv(128, 128),
            single_conv(64, 64)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128),
            # single_conv(256, 256),
            # single_conv(256, 256),
            single_conv(128, 128)
        )

        self.up1 = up(128)
        self.conv3 = nn.Sequential(
            single_conv(64, 64),
            # single_conv(128, 128),
            single_conv(64, 64)
        )

        self.up2 = up(64)
        self.conv4 = nn.Sequential(
            single_conv(32, 32),
            single_conv(32, 32)
        )

        self.outc = outconv(32, 1)


        # CBDNet_00
        # self.inc = nn.Sequential(
        #     single_conv(6, 32),
        #     single_conv(32, 32)
        # )
        #
        # self.down1 = nn.AvgPool2d(2)
        # self.conv1 = nn.Sequential(
        #     single_conv(32, 64),
        #     # single_conv(128, 128),
        #     single_conv(64, 64)
        # )
        #
        # self.down2 = nn.AvgPool2d(2)
        # self.conv2 = nn.Sequential(
        #     single_conv(64, 128),
        #     single_conv(128, 128),
        #     # single_conv(256, 256),
        #     # single_conv(256, 256),
        #     # single_conv(256, 256),
        #     single_conv(128, 128)
        # )
        #
        # self.up1 = up(128)
        # self.conv3 = nn.Sequential(
        #     single_conv(64, 64),
        #     # single_conv(128, 128),
        #     single_conv(64, 64)
        # )
        #
        # self.up2 = up(64)
        # self.conv4 = nn.Sequential(
        #     single_conv(32, 32),
        #     single_conv(32, 32)
        # )
        #
        # self.outc = outconv(32, 3)

        # CBDNet_mini
        # self.inc = nn.Sequential(
        #     single_conv(6, 64),
        #     single_conv(64, 64)
        # )
        #
        # self.down1 = nn.AvgPool2d(2)
        # self.conv1 = nn.Sequential(
        #     single_conv(64, 128),
        #     # single_conv(128, 128),
        #     single_conv(128, 128)
        # )
        #
        # self.down2 = nn.AvgPool2d(2)
        # self.conv2 = nn.Sequential(
        #     single_conv(128, 256),
        #     single_conv(256, 256),
        #     # single_conv(256, 256),
        #     # single_conv(256, 256),
        #     # single_conv(256, 256),
        #     single_conv(256, 256)
        # )
        #
        # self.up1 = up(256)
        # self.conv3 = nn.Sequential(
        #     single_conv(128, 128),
        #     # single_conv(128, 128),
        #     single_conv(128, 128)
        # )
        #
        # self.up2 = up(128)
        # self.conv4 = nn.Sequential(
        #     single_conv(64, 64),
        #     single_conv(64, 64)
        # )
        #
        # self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.fcn = FCN()
        # self.net1 = NET1()

        # self.SB = noSB()
        self.SB = SB()
        # self.unet = UNet()
        self.RENet = ResEnhanceNet()

        self.filter = filter()

        # self.fuse = noFuse()
        self.fuse = AdaptiveFuse()
    
    def forward(self, x):
        # noise_level = self.fcn(x)
        # noise_level = x

        # concat_img = torch.cat([x, noise_level], dim=1)
        # out = self.unet(concat_img) + x

        xh, xl = self.filter(x)
        xh = torch.cat([xh, x], 1)
        xl = torch.cat([xl, x], 1)

        # xh = torch.cat([x, x], 1)
        # xl = torch.cat([x, x], 1)

        xh = self.SB(xh)
        xl = self.RENet(xl)
       # xh = self.unet(xh)
        # x_fused, alpha_h, alpha_l = self.fuse(xh, xl)
        x_fused = self.fuse(xh, xl)

        return x_fused


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, out_image, gt_image):
        # 实现简易的 SSIM 计算
        mu_x = F.avg_pool2d(out_image, 3, 1)
        mu_y = F.avg_pool2d(gt_image, 3, 1)
        sigma_x = F.avg_pool2d(out_image ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(gt_image ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(out_image * gt_image, 3, 1) - mu_x * mu_y

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                    (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        ssim_loss = torch.clamp((1 - ssim) / 2, 0, 1)

        return ssim_loss.mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()  # 只使用 VGG 的前几层特征
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, out_image, gt_image):
        out_features = self.vgg(out_image)
        gt_features = self.vgg(gt_image)
        perceptual_loss = F.mse_loss(out_features, gt_features)

        return perceptual_loss

class fixed_loss(nn.Module):
    def __init__(self, alpha=1, beta=0.5, gamma=1):
        super().__init__()
        self.alpha = alpha  # 用于调节 SSIM Loss 的权重
        # self.beta = beta    # 用于调节 Perceptual Loss 的权重
        self.gamma = gamma  # 用于调节 L2 Loss 的权重
        self.ssim_loss_fn = SSIMLoss()
        # self.perceptual_loss_fn = PerceptualLoss()

    def forward(self, out_image, gt_image, gt_noise, if_asym):
        # L2 Loss (MSE)
        mse_loss = F.mse_loss(out_image, gt_image)

        # SSIM Loss
        ssim_loss = self.ssim_loss_fn(out_image, gt_image)

        # Perceptual Loss
        # out_image_3ch = out_image.repeat(1, 3, 1, 1)
        # gt_image_3ch = gt_image.repeat(1, 3, 1, 1)
        # perceptual_loss = self.perceptual_loss_fn(out_image_3ch, gt_image_3ch)

        # 综合损失函数
        # total_loss = (self.alpha * ssim_loss) + (self.beta * perceptual_loss) + (self.gamma * mse_loss)
        total_loss = (self.alpha * ssim_loss) + (self.gamma * mse_loss)
        # asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))
        #
        # h_x = est_noise.size()[2]
        # w_x = est_noise.size()[3]
        # count_h = self._tensor_size(est_noise[:, :, 1:, :])
        # count_w = self._tensor_size(est_noise[:, :, : ,1:])
        # h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        # w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        # tvloss = h_tv / count_h + w_tv / count_w
        #
        # loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss


        return total_loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]