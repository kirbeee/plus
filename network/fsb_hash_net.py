"""
@author: Jun Wang 
@date: 20201019
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from network import load_model
import numpy as np


#### Feature Extractor ####
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class Self_Att(nn.Module):
    def __init__(self, channels, num_heads):
        super(Self_Att, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class Feed_Forward(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(Feed_Forward, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class PFE_Block(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(PFE_Block, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = Self_Att(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = Feed_Forward(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        
        return x

# Main PFE Feature Extractor Network 
class FSB_Hash_Net(Module):   
    def __init__(self, embedding_size=1024, do_prob=0.0, out_h=7, out_w=7):
        super(FSB_Hash_Net, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))        
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.dropout = Dropout(do_prob)
        self.encoder_1 = PFE_Block(channels=64, num_heads=8, expansion_factor=2.66)
        self.encoder_2 = PFE_Block(channels=64, num_heads=8, expansion_factor=2.66)
        self.encoder_3 = PFE_Block(channels=128, num_heads=8, expansion_factor=2.66)
    
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
    
#### Hash Generator ####
# Random Shifting
class RandomShift(Module):
    def __init__(self, embedding_size, device):
        super(RandomShift, self).__init__()
        self.learnable_M = nn.Parameter(torch.rand(embedding_size, embedding_size))
        self.device = device
        self.FC1 = Linear(embedding_size, embedding_size)

    def forward(self, x, label):
        x_out = torch.tensor(()).to(self.device)

        for i in range(0, x.shape[0]):
            torch.manual_seed(label[i])
            torch.cuda.manual_seed_all(label[i])
            M_matrix = torch.randn((self.learnable_M.shape)).to(self.device)
            h1 = torch.diagonal(M_matrix)
            h = (x[i] + 0.01*(h1))
            x_out = torch.cat((x_out, h.unsqueeze(0)))
        out = self.FC1(x_out)
        return out

# Random Permutation
class RandomPerm(nn.Module):
    def __init__(self, device='cuda:0'):
        super(RandomPerm, self).__init__()
        self.device = device

    def forward(self, x, y):
        batch_n = x.shape[0]
        output_tensor = torch.tensor(()).to(self.device)
        for i in range(0, batch_n):            
            with torch.no_grad():
                torch.manual_seed(y[i])
                torch.cuda.manual_seed_all(y[i])
                perm_index = torch.randperm(x[i].shape[0])
                perm_x = x[i][perm_index].unsqueeze(0)
            output_tensor = torch.cat((output_tensor, perm_x), dim=0)
        return output_tensor

# Main Hash Generator Network
class Hash_Generator(Module):
    def __init__(self, embedding_size = 1024, do_prob = 0.0, device='cuda:0', out_embedding_size = 512):
        super(Hash_Generator, self).__init__()
        self.fc_hash = nn.Sequential(
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, out_embedding_size),                        
        ).to(device)
        self.Learnable_EX_hashingNet = RandomShift(out_embedding_size, device)
        self.bn = BatchNorm1d(out_embedding_size)
        self.dropout = Dropout(do_prob)        
        self.rand_pm = RandomPerm(device)
        self.device = device
    
    def forward(self, x, y=None, training=False):
        x = self.rand_pm(x, y) # user-specific permutation
        emb = self.fc_hash(x)
        emb = self.Learnable_EX_hashingNet(emb, y) # user-specific shifting
        emb = self.bn(emb)
        emb = self.dropout(emb)
        
        if training==True:
            return torch.tanh(F.normalize(emb, p=2, dim=1))
        else:
            return torch.sign(F.normalize(emb, p=2, dim=1))


#### Modality Discriminator ####
# Gradient Reversal Layer (GRL)
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

# Main Modality Discriminator Network
class Modality_Discriminator(Module):
  def __init__(self, input_dim=1024, hidden_size=512):
    super(Modality_Discriminator, self).__init__()
    self.ad_layer1 = Linear(input_dim, hidden_size)
    self.ad_layer2 = Linear(hidden_size, hidden_size)
    self.ad_layer3 = Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = Dropout(0.5)
    self.dropout2 = Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))   
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)   
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


if __name__ == '__main__':
    device = torch.device('cuda:0')
    load_model_path_fe = './models/best_feature_extractor/FSB-HashNet.pth'
    feature_extractor = FSB_Hash_Net(embedding_size = 1024, do_prob = 0.0).eval().to(device)
    feature_extractor = load_model.load_pretrained_network(feature_extractor, load_model_path_fe, device = device)
    
    load_model_path = load_model_path_fe.replace('best_feature_extractor', 'best_generator')
    hash_generator = Hash_Generator(embedding_size = 1024, device=device, out_embedding_size=512).eval().to(device)
    hash_generator = load_model.load_pretrained_network(hash_generator, load_model_path, device = device)