import torch
import torch.nn as nn
from torchjpeg import dct
from torch.nn import functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.left_conv_1 = ConvBlock(in_channels=in_channels, middle_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)

        self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2,
                                           output_padding=1)
        self.right_conv_2 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256)

        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2,
                                           output_padding=1)
        self.right_conv_3 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128)

        self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1,
                                           padding=1)
        self.right_conv_4 = ConvBlock(in_channels=128, middle_channels=64, out_channels=64)
        self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def encode(self, x):
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(feature_3_pool)
        feature_4_pool = self.pool_4(feature_4)

        feature_5 = self.left_conv_5(feature_4_pool)

        return feature_1, feature_2, feature_3, feature_4, feature_5

    def decode(self, feature_1, feature_2, feature_3, feature_4, feature_5, extract_features=False):
        de_feature_1 = self.deconv_1(feature_5)
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)

        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        out = self.right_conv_5(de_feature_4_conv)

        if extract_features:
            return out, (de_feature_1_conv, de_feature_2_conv, de_feature_3_conv, de_feature_4_conv)
        else:
            return out

    def forward(self, x):
        feature_1, feature_2, feature_3, feature_4, feature_5 = self.encode(x)
        out = self.decode(feature_1, feature_2, feature_3, feature_4, feature_5)

        return out


def dct_transform(x, chs_remove=None, chs_pad=False,
                  size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        Transform a spatial image into its frequency channels.
        Prune low-frequency channels if necessary.
    """

    # assert x is a (3, H, W) RGB image
    assert x.shape[1] == 3

    # convert the spatial image's range into [0, 1], recommended by TorchJPEG
    x = x * 0.5 + 0.5

    # up-sample
    x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)

    # convert to the YCbCr color domain, required by DCT
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128

    # 3. 執行 Block DCT (修正 view 邏輯)
    b, c, h, w = x.shape
    h_block, w_block = h // stride, w // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, h_block, w_block, size * size).permute(0, 1, 4, 2, 3)

    # prune channels
    if chs_remove is not None:
        channels = list(set([i for i in range(64)]) - set(chs_remove))
        if not chs_pad:
            # simply remove channels
            x_freq = x_freq[:, :, channels, :, :]
        else:
            # pad removed channels with zero, helpful for visualization
            x_freq[:, :, channels] = 0

    # 5. 堆疊 Y, Cb, Cr 頻道 -> (B, 192, H_block, W_block)
    x_freq = x_freq.reshape(b, -1, h_block, w_block)

    return x_freq


def idct_transform(x, size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        將 192 通道的頻域數據還原為空間域影像
    """
    b, ch_total, h_freq, w_freq = x.shape

    # 1. 還原為區塊格式 (B, 3, 64, H_f, W_f)
    x = x.view(b, 3, 64, h_freq, w_freq)
    x = x.permute(0, 1, 3, 4, 2)  # -> (B, 3, H_f, W_f, 64)
    x = x.view(b, 3, h_freq * w_freq, 8, 8)

    # 2. 執行 Block IDCT
    x_spatial = dct.block_idct(x)

    # 3. 重新拼湊像素塊 (修正 view 與 permute)
    x_spatial = x_spatial.view(b, 3, h_freq, w_freq, 8, 8)
    x_spatial = x_spatial.permute(0, 1, 2, 4, 3, 5).contiguous()
    x_spatial = x_spatial.view(b, 3, h_freq * 8, w_freq * 8)

    # 4. 色彩空間還原 (YCbCr -> RGB)
    x_spatial = x_spatial + 128
    x_spatial = dct.to_rgb(x_spatial)
    x_spatial = x_spatial / 255.0

    # 5. 縮小回原始比例 (如果當初有放大)
    if ratio != 1:
        x_spatial = F.interpolate(x_spatial, scale_factor=1 / ratio, mode='bilinear', align_corners=True)

    return x_spatial

