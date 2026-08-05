import math
import torch
import torch.nn.functional as F
import torch.nn as nn
# from einops import repeat, rearrange
from .vmamba import SS2D

def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)

def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)

def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

class CrossScan(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # Reverse the reversed horizontal and vertical parts and add them to the original horizontal and vertical parts
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # Convert the vertical part into horizontal part, add them together, and then convert them back to the original matrix form.
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # Reverse the reverse part of the oblique and counter-oblique directions and add them to the original oblique and counter-oblique directions
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # Convert the oblique and anti-oblique parts into their original matrix form and add them together
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res

class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # Convert the vertical part into horizontal part, add them together, and then convert them back to the original matrix form.
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # Reverse the reverse part of the oblique and counter-oblique directions and add them to the original oblique and counter-oblique directions
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # Convert the oblique and anti-oblique parts into their original matrix form and add them together
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # Horizontal and vertical scanning
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # oblique and reverse oblique scanning
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)
# SSM arous
class EfficientScan(torch.autograd.Function):
    # [B, C, H, W] -> [B, 4, C, H * W] (original)
    # [B, C, H, W] -> [B, 4, C, H/w * W/w]
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size=2):  # [B, C, H, W] -> [B, 4, C, H/w * W/w]
        B, C, org_h, org_w = x.shape
        ctx.shape = (B, C, org_h, org_w)
        ctx.step_size = step_size

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, 4, C, H * W))

        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        xs = xs.view(B, 4, C, -1)
        return xs

    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor):  # [B, 4, H/w * W/w] -> [B, C, H, W]

        B, C, org_h, org_w = ctx.shape
        step_size = ctx.step_size

        newH, newW = math.ceil(org_h / step_size), math.ceil(org_w / step_size)
        grad_x = grad_xs.new_empty((B, C, newH * step_size, newW * step_size))

        grad_xs = grad_xs.view(B, 4, C, newH, newW)

        grad_x[:, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)
        grad_x[:, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)

        if org_h != grad_x.shape[-2] or org_w != grad_x.shape[-1]:
            grad_x = grad_x[:, :, :org_h, :org_w]

        return grad_x, None

# class EfficientMerge(torch.autograd.Function):  # [B, 4, C, H/w * W/w] -> [B, C, H*W]
#     @staticmethod
#     def forward(ctx, ys: torch.Tensor, ori_h: int, ori_w: int, step_size=2):
#         B, K, C, L = ys.shape
#         H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)
#         ctx.shape = (H, W)
#         ctx.ori_h = ori_h
#         ctx.ori_w = ori_w
#         ctx.step_size = step_size
#
#         new_h = H * step_size
#         new_w = W * step_size
#
#         y = ys.new_empty((B, C, new_h, new_w))
#
#         y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, W)
#         y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
#         y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, W)
#         y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
#
#         if ori_h != new_h or ori_w != new_w:
#             y = y[:, :, :ori_h, :ori_w].contiguous()
#
#         y = y.view(B, C, -1)
#         return y
#
#     @staticmethod
#     def backward(ctx, grad_x: torch.Tensor):  # [B, C, H*W] -> [B, 4, C, H/w * W/w]
#
#         H, W = ctx.shape
#         B, C, L = grad_x.shape
#         step_size = ctx.step_size
#
#         grad_x = grad_x.view(B, C, ctx.ori_h, ctx.ori_w)
#
#         if ctx.ori_w % step_size != 0:
#             pad_w = step_s

class WarpedSS2D(nn.Module):
    SPLIT_RATIOS = [1 / 4, 1 / 2, 1 / 2, 3 / 4]

    def __init__(self, d_model, d_state=8, d_conv=7, expand=1, drop_rate=0., ssm_ratio=2.0, index=2):
        super().__init__()
        self.index = index
        self.use_pool = (index < 3)
        self.pool_scale = 2 ** (3 - index)

        d_expand = int(ssm_ratio * d_model)
        d_inner1 = int(d_expand * self.SPLIT_RATIOS[index])
        d_inner2 = d_expand - d_inner1
        self.split = (d_inner1, d_inner2)

        self.in_proj = nn.Sequential(
            nn.Conv2d(d_model, d_expand, kernel_size=1),
            nn.GELU(),
        )

        self.local_conv = nn.Sequential(
            nn.Conv2d(d_inner2, d_inner2, kernel_size=7, stride=1, padding=3, groups=d_inner2),
            nn.BatchNorm2d(d_inner2),
        )

        self.atn = SS2D(d_model=d_inner1, d_state=d_state, d_conv=d_conv, expand=expand)

        self.out_proj = nn.Sequential(
            nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=self.pool_scale) if self.use_pool else nn.Identity()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.in_proj(x)
        x_low, x_high = torch.split(x, self.split, dim=1)
        x_high = self.local_conv(x_high)

        if self.use_pool:
            x_low_orig = x_low
            x_low_pooled = self.avgpool(x_low)
            high_freq = x_low_orig - F.interpolate(x_low_pooled, size=(H, W), mode='nearest')
            attn_out = self.atn(x_low_pooled)
            x_low = F.interpolate(attn_out, scale_factor=self.pool_scale, mode='bilinear',
                                  align_corners=False) + high_freq
        else:
            x_low = self.atn(x_low)

        x = torch.cat((x_low, x_high), dim=1)
        return self.dropout(self.out_proj(x))