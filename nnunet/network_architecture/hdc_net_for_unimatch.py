import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm3d

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


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params

        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.activation =self.params['activation']
    
        self.conv_3x3x3 = Conv_3x3x3(
            self.in_chns*8 , self.ft_chns[0], self.activation)
        self.conv1 = HDC_module(
            self.ft_chns[0], self.ft_chns[0], self.activation)
        self.down1 = Conv_down(
            self.ft_chns[0], self.ft_chns[1], self.activation)
        self.conv2 = HDC_module(
            self.ft_chns[1], self.ft_chns[1], self.activation)
        self.down2 = Conv_down(
            self.ft_chns[1], self.ft_chns[2], self.activation)
        self.conv3 = HDC_module(
            self.ft_chns[2], self.ft_chns[2], self.activation)
        self.down3 = Conv_down(
            self.ft_chns[2], self.ft_chns[3], self.activation)
        
    @autocast()
    def forward(self, x): # 128
        x = hdc(x)      # 64
        x0 = self.conv_3x3x3(x) 
        x = self.conv1(x0)  
        x1 = self.down1(x) #32

        x2 = self.conv2(x1)
        x2 = self.down2(x2) # 16

        x3 = self.conv3(x2)
        x3 = self.down3(x3) # 8

        return [x0, x1, x2, x3 ]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params

        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.activation = self.params['activation']
        
        self.bridge = HDC_module(self.ft_chns[3], self.ft_chns[3], self.activation)
        self.up_1 = conv_trans_block_3d(self.ft_chns[3], self.ft_chns[3], self.activation)
        self.conv_1 = HDC_module(self.ft_chns[3]+self.ft_chns[2], self.ft_chns[2], self.activation)
        self.up_2 = conv_trans_block_3d(self.ft_chns[2], self.ft_chns[2], self.activation)
        self.conv_2 = HDC_module(self.ft_chns[2]+self.ft_chns[1], self.ft_chns[1], self.activation)
        self.up_3 = conv_trans_block_3d(self.ft_chns[1], self.ft_chns[1], self.activation)
        self.conv_3 = HDC_module(self.ft_chns[1]+self.ft_chns[0], self.ft_chns[0], self.activation)
        #self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample = conv_trans_block_3d(self.ft_chns[0], self.ft_chns[0], self.activation)
        self.out = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=1, stride=1, padding=0)
        
    @autocast()
    def forward(self, feature):
        x1 = feature[0] # 64
        x2 = feature[1] # 32 
        x3 = feature[2] # 16
        x4 = feature[3] # 8

        x= self.bridge(x4)
        x = self.up_1(x) # 16
        x = torch.cat((x, x3), dim=1)
        x= self.conv_1(x)
        x = self.up_2(x) # 32
        x = torch.cat((x, x2), dim=1)
        x= self.conv_2(x)
        x = self.up_3(x) # 64
        x = torch.cat((x, x1), dim=1)
        x = self.conv_3(x)
        x = self.upsample(x) # 128
        x = self.out(x)

        return x
        

class HDC_Net(nn.Module):
    """
    一中轻量级的网络，是用了HDC模块简化Conv3D的计算量和参数，同时提出了一个具有固定中间通道数的u-net类型的网络
    """
    def __init__(self, in_chns,class_num,feature_chns,acti_func="leak_relu"):
        super(HDC_Net, self).__init__()

        if acti_func == 'relu':
            activation = nn.ReLU(inplace=False)
        elif acti_func == 'leak_relu':
            activation = nn.LeakyReLU(inplace=False)
        else:
            activation = nn.ReLU(inplace=True)

        params = {'in_chns': in_chns,
                  'feature_chns': feature_chns,
                  'class_num': class_num,
                  'activation': activation}
        
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    @autocast()
    def forward(self, x, need_fp=False):
        feature = self.encoder(x)
        
        if need_fp:
            outs = self.decoder([torch.cat((feat, nn.Dropout3d(0.5)(feat))) for feat in feature])
            return outs.chunk(2)
        
        output = self.decoder(feature)
        return output
    