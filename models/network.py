import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import torchvision
from .prompt import F_ext_bri as F_ext_bri
from .prompt import F_ext_content as F_ext_content
# Encoder Block
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, mode):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, mode, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Decoder Block
class DBlock(nn.Module):
    def __init__(self, channel, num_res, mode):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class Network(nn.Module):
    def __init__(self, mode, num_res=8):
        super(Network, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, mode),
            EBlock(base_channel*2, num_res, mode),
            EBlock(base_channel*4, num_res, mode),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, mode),
            DBlock(base_channel * 2, num_res, mode),
            DBlock(base_channel, num_res, mode)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.F_ext_bri = F_ext_bri(middle_channel=40)
        self.F_ext_content = F_ext_content(middle_channel=3)
        self.Pro_FAM1 = FAM(base_channel * 2)
        self.Pro_FAM2 = FAM(base_channel * 4)
        self.Pro_FAM3 = FAM(base_channel)

    def forward(self, x, pro_img=None):
        x_adjust = x
        if pro_img is not None:
            illu_pro = self.F_ext_bri(pro_img)
                # torchvision.utils.save_image(illu_map, 'results/illu_map.jpg')
            x_adjust = x * illu_pro + x
            content_pro = self.F_ext_content(pro_img)
            pro_res1 = self.feat_extract[0](content_pro)
            pro_res2 = self.feat_extract[1](pro_res1)
            pro_res3 = self.feat_extract[2](pro_res2)

        x_2 = F.interpolate(x_adjust, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256*256
        x_ = self.feat_extract[0](x_adjust)
        res1 = self.Encoder[0](x_)

        # 128*128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        # 64*64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        if pro_img is not None:
            z = self.Pro_FAM2(z, pro_res3)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128*128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        if pro_img is not None:
            res2 = self.Pro_FAM1(res2, pro_res2)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256*256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        if pro_img is not None:
            res1 = self.Pro_FAM3(res1, pro_res1)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs

