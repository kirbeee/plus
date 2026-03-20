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

    # perform block discrete cosine transform (BDCT)
    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

    # prune channels
    if chs_remove is not None:
        channels = list(set([i for i in range(64)]) - set(chs_remove))
        if not chs_pad:
            # simply remove channels
            x_freq = x_freq[:, :, channels, :, :]
        else:
            # pad removed channels with zero, helpful for visualization
            x_freq[:, :, channels] = 0

    # stack frequency channels from each color domain
    x_freq = x_freq.reshape(b, -1, n_block, n_block)

    # 【新增】將 DCT 係數正規化 (除以 255 或 100 都可以，這裡以 255 為例)
    # 這樣會讓特徵數值回到接近 UNet 好初始化的範圍
    x_freq = x_freq / 255.0

    return x_freq


def idct_transform(x, size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        The inverse of DCT transform.
        Transform frequency channels (must be 192 channels, can be padded with 0) back to the spatial image.
    """

    # 【新增】還原 DCT 係數的尺度
    x = x * 255.0

    b, _, h, w = x.shape

    x = x.view(b, 3, 64, h, w)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(b, 3, h * w, 8, 8)
    x = dct.block_idct(x)
    x = x.view(b * 3, h * w, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * ratio, 112 * ratio),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(b, 3, 112 * ratio, 112 * ratio)
    x = x + 128
    x = dct.to_rgb(x)
    x = x / 255
    x = F.interpolate(x, scale_factor=1 / ratio, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x

