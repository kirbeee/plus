"""
Mamba_Hash_Net
==============

Same overall topology as `FSB_Hash_Net` (network/fsb_hash_net.py), except the
"bottleneck" stages -- originally `Residual` blocks made of stacked
`Depth_Wise(residual=True)` inverted-residual units -- are replaced with
stacks of `WarpedSS2D` (network/mamba_logits.py), a Mamba/SS2D-based block
that mixes a 2D selective-scan branch (on a pooled low-frequency channel
split) with a depthwise-conv branch (on a high-frequency channel split).

Everything else (stem, stride-2 down-sampling transitions, the PFE_Block
self-attention encoders inserted between stages, and the embedding head)
is kept identical to FSB_Hash_Net so weights/shapes stay drop-in compatible
with the rest of the training pipeline (001_train.py, fsb_logits.py, etc).

IMPORTANT DEPENDENCY NOTE
--------------------------
`WarpedSS2D` (in mamba_logits.py) instantiates `SS2D(...)`, but `SS2D` is
not defined or imported anywhere in the mamba_logits.py that was provided.
Make sure `SS2D` is available in that module's namespace (e.g. add
`from .vmamba import SS2D` or paste the class definition in) before using
Mamba_Hash_Net, otherwise construction will raise `NameError: SS2D`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsb_hash_net import Conv_block, Linear_block, Depth_Wise, Flatten, PFE_Block
from .mamba_logits import WarpedSS2D


class MambaBottleneck(nn.Module):
    """Stack of `num_block` WarpedSS2D blocks.

    Drop-in replacement for `Residual` (stack of `Depth_Wise(residual=True)`
    blocks) in FSB_Hash_Net -- same role (per-resolution feature refinement
    at constant channel count / spatial size), different mechanism (Mamba
    2D-scan + local conv instead of depthwise-separable conv).
    """

    def __init__(self, channels, num_block, index, d_state=8, d_conv=7,
                 expand=1, ssm_ratio=2.0, drop_rate=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            WarpedSS2D(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_rate=drop_rate,
                ssm_ratio=ssm_ratio,
                index=index,
            )
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.blocks(x)


class Mamba_Hash_Net(nn.Module):
    """Feature extractor used by 001_train.py in place of FSB_Hash_Net.

    Args:
        embedding_size: output embedding dim (matches FSB_Hash_Net).
        do_prob: dropout prob for the head and (optionally) each
            WarpedSS2D block.
        out_h, out_w: spatial size fed into conv_6_dw's kernel. Also used
            below to justify the WarpedSS2D `index` -> pool_scale mapping
            (assumes 112x112 input -> 7x7 at conv_5, i.e. out_h=out_w=7).
            If your input resolution differs, re-check that the pool_scale
            for each stage divides its feature-map size sensibly.
        d_state, d_conv, expand, ssm_ratio: SS2D/WarpedSS2D hyperparameters,
            forwarded to every bottleneck block.
    """

    def __init__(self, embedding_size=1024, do_prob=0.0, out_h=7, out_w=7,
                 d_state=8, d_conv=7, expand=1, ssm_ratio=2.0):
        super().__init__()

        # --- stem (unchanged) ---
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        # --- stride-2 down-sampling transitions (unchanged) ---
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)

        # --- bottleneck stages: Residual(Depth_Wise) -> MambaBottleneck(WarpedSS2D) ---
        # index chosen so every stage's internal SS2D attention runs on a
        # 7x7 map (assumes 112x112 input, out_h=out_w=7):
        #   conv_3 @ 28x28, pool_scale=2**(3-1)=4 -> 7x7
        #   conv_4 @ 14x14, pool_scale=2**(3-2)=2 -> 7x7
        #   conv_5 @  7x7,  index=3 -> no pooling, already 7x7
        self.conv_3 = MambaBottleneck(64, num_block=4, index=1,
                                       d_state=d_state, d_conv=d_conv,
                                       expand=expand, ssm_ratio=ssm_ratio,
                                       drop_rate=do_prob)
        self.conv_4 = MambaBottleneck(128, num_block=6, index=2,
                                       d_state=d_state, d_conv=d_conv,
                                       expand=expand, ssm_ratio=ssm_ratio,
                                       drop_rate=do_prob)
        self.conv_5 = MambaBottleneck(128, num_block=2, index=3,
                                       d_state=d_state, d_conv=d_conv,
                                       expand=expand, ssm_ratio=ssm_ratio,
                                       drop_rate=do_prob)

        # --- head (unchanged) ---
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(do_prob)

        # --- inter-stage self-attention encoders (unchanged) ---
        self.encoder_1 = PFE_Block(channels=64, num_heads=8, expansion_factor=2.66)
        self.encoder_2 = PFE_Block(channels=64, num_heads=8, expansion_factor=2.66)
        self.encoder_3 = PFE_Block(channels=128, num_heads=8, expansion_factor=2.66)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)

        out = self.encoder_1(out)
        out = self.conv_23(out)
        out = self.conv_3(out)          # Mamba bottleneck stage 1 (28x28, 64ch)

        out = self.encoder_2(out)
        out = self.conv_34(out)
        out = self.conv_4(out)          # Mamba bottleneck stage 2 (14x14, 128ch)

        out = self.encoder_3(out)
        out = self.conv_45(out)
        out = self.conv_5(out)          # Mamba bottleneck stage 3 (7x7, 128ch)

        out = self.conv_6_sep(out)
        emb = self.conv_6_dw(out)

        emb = self.conv_6_flatten(emb)
        emb = self.dropout(emb)
        emb = self.linear(emb)
        emb = self.bn(emb)

        return F.normalize(emb, p=2, dim=1)


if __name__ == '__main__':
    # quick shape sanity check (requires SS2D to be resolvable in mamba_logits.py,
    # and the mamba_ssm package installed with a CUDA build for selective_scan_fn)
    model = Mamba_Hash_Net(embedding_size=1024)
    x = torch.randn(2, 3, 112, 112)
    y = model(x)
    print(y.shape)  # expected: torch.Size([2, 1024])