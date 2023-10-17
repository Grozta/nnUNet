from typing import Tuple, Union, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn as nndsfadf3454
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm3d

from scipy.ndimage.filters import gaussian_filter
from batchgenerators.augmentations.utils import pad_nd_image
from tqdm import tqdm

from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.tensor_utilities import flip
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch



__all__ = ['HDC_Net']

class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.norm = BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)
        self.norm = BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_down, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        self.norm = BatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class HDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(HDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_2 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_3 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_3x3x1_1(x2)
        x3 = self.conv_3x3x1_2(x2 + x3)
        x4 = self.conv_3x3x1_3(x3 + x4)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1


class DUC(nn.Module):

    def __init__(self, upscale_factor, class_num, in_channels):
        """
        reference paper: Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016).
         Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network.
          In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).
        3D DUC module, the input data dimensions should be 5D tensor like(batch, channel, x, y, z),
        workflow: conv->batchnorm->relu->pixelshuffle
        :param upscale_factor(int, tuple, list): Scale up factor, if the input scale_factor is int,
         x,y,z axes of a data will scale up with the same scale factor,
         else x,y,z axes of a data will scale with different scale factor
        :param class_num(int): the number of total classes (background and instance)
        :param in_channels(int): the number of input channel
        """
        super(DUC, self).__init__()
        if isinstance(upscale_factor, int):
            self.scale_factor_x = self.scale_factor_y = self.scale_factor_z = upscale_factor
        elif isinstance(upscale_factor, tuple) or isinstance(upscale_factor, list):
            self.scale_factor_x = upscale_factor[0]
            self.scale_factor_y = upscale_factor[1]
            self.scale_factor_z = upscale_factor[2]
        else:
            print("scale factor should be int or tuple")
            raise ValueError
        # self.conv = nn.Conv3d(in_channels, class_num * scale_factor_x * scale_factor_y * scale_factor_z, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)
        self.conv = HDC_module(in_channels, class_num * self.scale_factor_x * self.scale_factor_y * self.scale_factor_z, self.activation)
        self.bn = nn.BatchNorm3d(class_num * self.scale_factor_x * self.scale_factor_y * self.scale_factor_z)
        self.relu = nn.ReLU(inplace=True)
    
    def pixelshuffle3d(self,x: torch.Tensor):
        pD = self.scale_factor_x
        pH = self.scale_factor_y
        pW = self.scale_factor_z
        B, iC, iD, iH, iW = x.shape

        oC, oD, oH, oW = iC//(pH*pW*pD),iD*pD, iH*pH, iW*pW
        x = x.reshape(B, oC, pD, pH, pW, iD, iH, iW)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)     # B, oC, iH, pH, iW, pW
        x = x.reshape(B, oC, oD, oH, oW)
        return x

    def pixelshuffle3d_invert(self, x: torch.Tensor):
        pD = self.scale_factor_x
        pH = self.scale_factor_y
        pW = self.scale_factor_z
        y = x
        B, iC, iD, iH, iW = y.shape
        oC, oD, oH, oW = iC*(pH*pW*pD), iD//pD,  iH//pH, iW//pW
        y = y.reshape(B, iC, oD, pD,oH, pH, oW, pW)
        y = y.permute(0, 1, 3, 5, 7, 2, 4, 6)     # B, iC, pH, pW, oH, oW
        y = y.reshape(B, oC, oD, oH, oW)
        return y

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixelshuffle3d(x)
        return x


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        BatchNorm3d(out_dim),
        activation)
    
device1 = torch.device("cuda")
def hdc(image, step=2):
        x1 = torch.Tensor([]).to(device1)
        for i in range(step):
            for j in range(step):
                for k in range(step):
                    x3 = image[:, :, k::step, i::step, j::step]
                    x1 = torch.cat((x1, x3), dim=1) 
        del image, x3
        torch.cuda.empty_cache()
        return x1


class HDC_Net(SegmentationNetwork):
    """
    一中轻量级的网络，是用了HDC模块简化Conv3D的计算量和参数，同时提出了一个具有固定中间通道数的u-net类型的网络
    """
    def __init__(self, 
                 in_dim=1,
                 layer_channels=[128,256,256,512],
                 out_dim=13,
                 semi_supervised=False,
                 is_dynamic_empty_cache=True):
        super().__init__()
        # HDC_Net parameter.
        self.semi_supervised = semi_supervised
        self.in_dim =in_dim
        self.out_dim = out_dim
        # self.raw_input_size = input_size
        self.n_f = layer_channels
        
        # self._check_input_size_avaiable()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache if not self.semi_supervised else False
        self.activation = nn.ReLU(inplace=False)

        self.conv_3x3x3 = Conv_3x3x3(self.in_dim*8 , self.n_f[0], self.activation)
        self.conv_1 = HDC_module(self.n_f[0], self.n_f[0], self.activation)
        self.down_1 = Conv_down(self.n_f[0], self.n_f[1], self.activation)
        self.conv_2 = HDC_module(self.n_f[1], self.n_f[1], self.activation)
        self.down_2 = Conv_down(self.n_f[1], self.n_f[2], self.activation)
        self.conv_3 = HDC_module(self.n_f[2], self.n_f[2], self.activation)
        self.down_3 = Conv_down(self.n_f[2], self.n_f[3], self.activation)
        # bridge
        self.bridge = HDC_module(self.n_f[3], self.n_f[3], self.activation)
        # 将桥从hdc_module换成3d卷积
        #self.bridge = Conv_3x3x3(self.n_f[3], self.n_f[3], self.activation)
        # up
        self.up_1 = conv_trans_block_3d(self.n_f[3], self.n_f[3], self.activation)
        self.conv_4 = HDC_module(self.n_f[3]+self.n_f[2], self.n_f[2], self.activation)
        self.up_2 = conv_trans_block_3d(self.n_f[2], self.n_f[2], self.activation)
        self.conv_5 = HDC_module(self.n_f[2]+self.n_f[1], self.n_f[1], self.activation)
        self.up_3 = conv_trans_block_3d(self.n_f[1], self.n_f[1], self.activation)
        self.conv_6 = HDC_module(self.n_f[1]+self.n_f[0], self.n_f[0], self.activation)

        #self.duc = DUC(2,self.out_dim,self.n_f[0])
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.out = nn.Conv3d(self.n_f[0], self.out_dim, kernel_size=1, stride=1, padding=0)
        
        self._initialize_weights()

    def _check_input_size_avaiable(self,x):
        """检查输入尺寸是否可用"""
        resolution = (x.shape)[-3:]
        for size in resolution:
            remain = 2**(len(self.n_f))
            if size % (remain)!= 0 :
                raise ValueError("Input size:{} is disable, must {} multiple".format(size,remain)) 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    @autocast()
    def forward(self, x):
        self._check_input_size_avaiable(x)
        x = hdc(x)
        x = self.conv_3x3x3(x)
        x1 = self.conv_1(x)
        x = self.down_1(x1)
        x2 = self.conv_2(x)
        x = self.down_2(x2)
        x3 = self.conv_3(x)
        x = self.down_3(x3)
        x = self.bridge(x)
        x = self.up_1(x)
        x = torch.cat((x, x3), dim=1)
        if self.is_dynamic_empty_cache:
            del x3
            torch.cuda.empty_cache()
        x = self.conv_4(x)
        x = self.up_2(x)
        x = torch.cat((x, x2), dim=1)
        if self.is_dynamic_empty_cache:
            del x2
            torch.cuda.empty_cache()
        x = self.conv_5(x)
        x = self.up_3(x)
        x = torch.cat((x, x1), dim=1)
        if self.is_dynamic_empty_cache:
            del x1
            torch.cuda.empty_cache()
        x = self.conv_6(x)
        
        #x = self.duc(x)
        x = self.upsample(x)
        x = self.out(x)

        return x
    
    @property
    def model_name(self):
        return "HDC_Net"
    