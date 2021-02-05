import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return [self.LL(x), self.LH(x), self.HL(x), self.HH(x)]

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

class CA(nn.Module):
    """
    通道注意力
    """

    def __init__(self, C, r):
        super(CA, self).__init__()
        #self.DreLU = DyReLUA(C // r)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // r, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(C // r, C, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        y = self.block(input)

        return input * y


class RCAB(nn.Module):
    def __init__(self, C, r):
        super(RCAB, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1, bias=True),
            CA(C, r)
        )

    def forward(self, input):
        y = self.block(input)
        return input + y

class RAC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RAC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            RCAB(out_ch, 8),
            RCAB(out_ch, 8)
        )

    def forward(self, input):
        return self.conv(input)


class Conv2d_Relu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv2d_Relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, input):
        return self.conv(input)


class LAST_U_net(nn.Module):
    def __init__(self, ):
        super(LAST_U_net, self).__init__()
        self.WP_Depth_1_2 = WavePool(1)
        self.WP_Depth_1_4 = WavePool(1)
        self.WP_Depth_1_8 = WavePool(1)
        self.WP_Depth_1_16 = WavePool(1)
        # self.P_p1_Depth = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.P_p2_Depth = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.P_p3_Depth = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.P_p4_Depth = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.P_c1_Depth = Conv2d_Relu(1, 64)
        self.P_c2_Depth = Conv2d_Relu(1, 128)
        self.P_c3_Depth = Conv2d_Relu(1, 256)
        self.P_c4_Depth = Conv2d_Relu(1, 512)

        self.WP_Haze_1_2 = WavePool(3)
        self.WP_Haze_1_4 = WavePool(3)
        self.WP_Haze_1_8 = WavePool(3)
        self.WP_Haze_1_16 = WavePool(3)
        self.P_c1_Haze = Conv2d_Relu(3, 64)
        self.P_c2_Haze = Conv2d_Relu(3, 128)
        self.P_c3_Haze = Conv2d_Relu(3, 256)
        self.P_c4_Haze = Conv2d_Relu(3, 512)

        self.conv1 = RAC(3, 64)
        self.pool1 = WavePool(64)
        self.conv2 = RAC(128, 128)
        self.pool2 = WavePool(128)
        self.conv3 = RAC(256, 256)
        self.pool3 = WavePool(256)
        self.conv4 = RAC(512, 512)
        self.pool4 = WavePool(512)
        self.conv5 = RAC(1024, 1024)

        self.convl1 = RAC(1, 64)
        self.pooll1 = WavePool(64)
        self.convl2 = RAC(128, 128)
        self.pooll2 = WavePool(128)
        self.convl3 = RAC(256, 256)
        self.pooll3 = WavePool(256)
        self.convl4 = RAC(512, 512)
        self.pooll4 = WavePool(512)
        self.convl5 = RAC(1024, 1024)

        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU()
        )
        self.conv6 = RAC(1536, 512)
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU()
        )
        self.conv7 = RAC(768, 256)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU()
        )
        self.conv8 = RAC(384, 128)
        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU()
        )
        self.conv9 = RAC(192, 64)

        self.upl6 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU()
        )
        self.convl6 = RAC(1536, 512)
        self.upl7 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU()
        )
        self.convl7 = RAC(768, 256)
        self.upl8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU()
        )
        self.convl8 = RAC(384, 128)
        self.upl9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU()
        )
        self.convl9 = RAC(192, 64)
        self.conv20 = RAC(128, 64)
        self.conv21 = Conv2d_Relu(64, 3)
        # self.BiLSTM512 = BILSTM(512, 512)
        # self.BiLSTM256 = BILSTM(256, 256)
        # self.BiLSTM128 = BILSTM(128, 128)
        # self.BiLSTM64 = BILSTM(64, 64)
        #self.waveunpool
    def forward(self, Depth, Haze):
        D_py1 = self.WP_Depth_1_2(Depth)[0]
        #D_py1 = self.P_p1_Depth(Depth)
        F_py1 = self.P_c1_Depth(D_py1)
        D_py2 = self.WP_Depth_1_4(D_py1)[0]
        F_py2 = self.P_c2_Depth(D_py2)
        D_py3 = self.WP_Depth_1_8(D_py2)[0]
        F_py3 = self.P_c3_Depth(D_py3)
        D_py4 = self.WP_Depth_1_16(D_py3)[0]
        F_py4 = self.P_c4_Depth(D_py4)

        Y_py1 = self.WP_Haze_1_2(Haze)[0]
        R_py1 = self.P_c1_Haze(Y_py1)
        Y_py2 = self.WP_Haze_1_4(Y_py1)[0]
        R_py2 = self.P_c2_Haze(Y_py2)
        Y_py3 = self.WP_Haze_1_8(Y_py2)[0]
        R_py3 = self.P_c3_Haze(Y_py3)
        Y_py4 = self.WP_Haze_1_16(Y_py3)[0]
        R_py4 = self.P_c4_Haze(Y_py4)

        Haze_c1 = self.conv1(Haze)
        Haze_p1 = self.pool1(Haze_c1)[0]
        Haze_p1 = torch.cat([Haze_p1, R_py1], dim=1)
        Haze_c2 = self.conv2(Haze_p1)
        Haze_p2 = self.pool2(Haze_c2)[0]
        Haze_p2 = torch.cat([Haze_p2, R_py2], dim=1)
        Haze_c3 = self.conv3(Haze_p2)
        Haze_p3 = self.pool3(Haze_c3)[0]
        Haze_p3 = torch.cat([Haze_p3, R_py3], dim=1)
        Haze_c4 = self.conv4(Haze_p3)
        Haze_p4 = self.pool4(Haze_c4)[0]
        Haze_p4 = torch.cat([Haze_p4, R_py4], dim=1)
        Haze_c5 = self.conv5(Haze_p4)

        Depth_c1 = self.convl1(Depth)
        Depth_p1 = self.pooll1(Depth_c1)[0]
        Depth_p1 = torch.cat([Depth_p1, F_py1], dim=1)
        Depth_c2 = self.convl2(Depth_p1)
        Depth_p2 = self.pooll2(Depth_c2)[0]
        Depth_p2 = torch.cat([Depth_p2, F_py2], dim=1)
        Depth_c3 = self.convl3(Depth_p2)
        Depth_p3 = self.pooll3(Depth_c3)[0]
        Depth_p3 = torch.cat([Depth_p3, F_py3], dim=1)
        Depth_c4 = self.convl4(Depth_p3)
        Depth_p4 = self.pooll4(Depth_c4)[0]
        Depth_p4 = torch.cat([Depth_p4, F_py4], dim=1)
        Depth_c5 = self.convl5(Depth_p4)

        Haze_up_6 = self.up6(Haze_c5)
        Depth_up_6 = self.upl6(Depth_c5)

        #Haze_up_6, Depth_up_6 = self.BiLSTM512(Haze_up_6, Depth_up_6)

        Haze_concat6 = torch.cat([Depth_up_6, Haze_up_6, Haze_c4], dim=1)
        Depth_concat6 = torch.cat([Haze_up_6, Depth_up_6, Depth_c4], dim=1)
        Haze_c6 = self.conv6(Haze_concat6)
        Depth_c6 = self.convl6(Depth_concat6)

        Haze_up_7 = self.up7(Haze_c6)
        Depth_up_7 = self.upl7(Depth_c6)

        #Haze_up_7, Depth_up_7 = self.BiLSTM256(Haze_up_7, Depth_up_7)

        Haze_concat7 = torch.cat([Depth_up_7, Haze_up_7, Haze_c3], dim=1)
        Depth_concat7 = torch.cat([Haze_up_7, Depth_up_7, Depth_c3], dim=1)
        Haze_c7 = self.conv7(Haze_concat7)
        Depth_c7 = self.convl7(Depth_concat7)

        Haze_up_8 = self.up8(Haze_c7)
        Depth_up_8 = self.upl8(Depth_c7)

        #Haze_up_8, Depth_up_8 = self.BiLSTM128(Haze_up_8, Depth_up_8)

        Haze_concat8 = torch.cat([Depth_up_8, Haze_up_8, Haze_c2], dim=1)
        Depth_concat8 = torch.cat([Haze_up_8, Depth_up_8, Depth_c2], dim=1)
        Haze_c8 = self.conv8(Haze_concat8)
        Depth_c8 = self.convl8(Depth_concat8)

        Haze_up_9 = self.up9(Haze_c8)
        Depth_up_9 = self.upl9(Depth_c8)

        #Haze_up_9, Depth_up_9 = self.BiLSTM64(Haze_up_9, Depth_up_9)

        Haze_concat9 = torch.cat([Depth_up_9, Haze_up_9, Haze_c1], dim=1)
        Depth_concat9 = torch.cat([Haze_up_9, Depth_up_9, Depth_c1], dim=1)
        Haze_c9 = self.convl9(Haze_concat9)
        Depth_c9 = self.convl9(Depth_concat9)

        c = self.conv20(torch.cat([Haze_c9, Depth_c9], dim=1))
        result = self.conv21(c)
        return result





