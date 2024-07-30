import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Borrowed from ''Improving image restoration by revisiting global information aggregation''
# --------------------------------------------------------------------------------
# train_size = (1, 3, 256, 256)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SpaBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=20, relu_slope=0.2):
        super(SpaBlock, self).__init__()
        self.spatialConv = nn.Sequential(*[
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False)
        ])
        self.identity = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x1):
        spatial_out = self.spatialConv(x1)
        identity_out = self.identity(x1)
        out = spatial_out + identity_out
        return out


# class FreBlock(nn.Module):
#     def __init__(self, in_channel=3, out_channel=20, relu_slope=0.2):
#         super(FreBlock, self).__init__()
#         self.fftConv2 = nn.Sequential(*[
#             nn.Conv2d(out_channel, out_channel, 1, 1, 0),
#             nn.LeakyReLU(relu_slope, inplace=False),
#             nn.Conv2d(out_channel, out_channel, 1, 1, 0)
#         ])
#
#     def forward(self, x1):
#         x_fft = torch.fft.rfft2(x1, norm='backward')
#         x_amp = torch.abs(x_fft)
#         x_phase = torch.angle(x_fft)
#         # enhanced_phase = self.fftConv2(x_phase)
#         enhanced_amp = self.fftConv2(x_amp)
#         out = torch.fft.irfft2(enhanced_amp * torch.exp(1j * x_phase), norm='backward')
#         return out

class FreBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=20, relu_slope=0.2):
        super(FreBlock, self).__init__()
        self.fftConv2 = nn.Sequential(*[
            nn.Conv2d(out_channel, out_channel, 1, 1, 0),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        ])

    def forward(self, x1):
        x_fft = torch.fft.rfft2(x1, norm='backward')
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)
        enhanced_phase = self.fftConv2(x_phase) + x_phase
        enhanced_amp = self.fftConv2(x_amp) + x_amp
        out = torch.fft.irfft2(enhanced_amp * torch.exp(1j * enhanced_phase), norm='backward')
        return out



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mode, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        # self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

        self.spaBlock = SpaBlock(in_channel, out_channel, relu_slope=0.2)
        self.freBlock = FreBlock(in_channel, out_channel, relu_slope=0.2)

        self.filter = filter

    def forward(self, x):
        out = self.conv1(x)
        #
        if self.filter:
            out = self.spaBlock(out)
            out = self.freBlock(out)

        out = out + x
        return out
