import torch.nn as nn
import torch.nn.functional as F
from .fsb_hash_net import Conv_block, Linear_block, Depth_Wise, Flatten,Feed_Forward
from .mamba_logits import WarpedSS2D

# TODO : rename this class -> this is not a bottleneck
class MambaBottleneck(nn.Module):
    def __init__(self, c, index,num_block=1, d_state=8, d_conv=7,
                 expand=1, ssm_ratio=2.0, drop_rate=0.0):
        super().__init__()
        modules = []
        for _ in range(num_block):
            modules.append(WarpedSS2D(d_model=c,d_state=d_state, d_conv=d_conv,expand=expand,drop_rate=drop_rate,ssm_ratio=ssm_ratio,index=index))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)


class Mamba_PFE_Block(nn.Module):
    def __init__(self, channels,  expansion_factor):
        super(Mamba_PFE_Block, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.mamba = WarpedSS2D(d_model=channels,d_state=16,d_conv=5)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = Feed_Forward(channels,expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))

        return x

class Mamba_Hash_Net(nn.Module):
    """
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

        # --- stem ---
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        # --- stride-2 down-sampling transitions ---
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = MambaBottleneck(64, num_block=4, index=1, d_state=d_state, d_conv=d_conv, expand=expand, ssm_ratio=ssm_ratio, drop_rate=do_prob)
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = MambaBottleneck(128, num_block=6, index=2,d_state=d_state, d_conv=d_conv, expand=expand, ssm_ratio=ssm_ratio, drop_rate=do_prob)
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = MambaBottleneck(128, num_block=2, index=3, d_state=d_state, d_conv=d_conv, expand=expand, ssm_ratio=ssm_ratio, drop_rate=do_prob)

        # --- head ---
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(do_prob)

        # --- inter-stage self-attention encoders ---
        self.encoder_1 = Mamba_PFE_Block(channels=64, expansion_factor=2.66)
        self.encoder_2 = Mamba_PFE_Block(channels=64, expansion_factor=2.66)
        self.encoder_3 = Mamba_PFE_Block(channels=128, expansion_factor=2.66)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)

        out = self.encoder_1(out)
        out = self.conv_23(out)
        out = self.conv_3(out)

        out = self.encoder_2(out)
        out = self.conv_34(out)
        out = self.conv_4(out)

        out = self.encoder_3(out)
        out = self.conv_45(out)
        out = self.conv_5(out)

        out = self.conv_6_sep(out)
        emb = self.conv_6_dw(out)

        emb = self.conv_6_flatten(emb)
        emb = self.dropout(emb)
        emb = self.linear(emb)
        emb = self.bn(emb)

        return F.normalize(emb, p=2, dim=1)