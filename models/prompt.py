import torch.nn as nn
import torch
from Utils.utils import *
import torch.nn.functional as F


class F_ext_bri(nn.Module):
    def __init__(self, in_channel=4, middle_channel=40, out_channel=3):
        super(F_ext_bri, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, middle_channel, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(middle_channel, middle_channel, kernel_size=5, padding=2, bias=True, groups=in_channel)
        self.conv2 = nn.Conv2d(middle_channel, out_channel, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_map


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)

        return x


class F_ext_content(nn.Module):
    def __init__(self, in_channel=3, middle_channel=32):
        super(F_ext_content, self).__init__()
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(middle_channel, middle_channel, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gradient = Get_gradient_nopadding()

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        out = self.gradient(conv2_out)

        return out

