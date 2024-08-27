import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from timm.models.layers import trunc_normal_
from model.blocks import CBlock_ln
from model.global_net import Global_pred



class Basenet(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Basenet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if type == 'ccc':
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        self.add_blocks = nn.Sequential(*blocks2)
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        add = self.add_blocks(img1) + img1
        add = self.add_end(add)

        return add


class Exd(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(Exd, self).__init__()
        numer_f = 3
        self.local_net = Basenet(in_dim=in_dim)
        self.local_net_1 = Basenet(in_dim=in_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)
        self.conv = nn.Conv2d(numer_f * 3, numer_f, 3, 1, 1)
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=1)
        self.nlt_weight = Parameter(torch.ones(1, 3, 1, 1))
        self.nlt_bias = Parameter(torch.zeros(1, 3, 1, 1))

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        img_low_raw = img_low[:, 0:3, :, :]
        img_low_exposure_1 = img_low[:, 3:6, :, :]
        img_low_exposure_2 = img_low[:, 6:9, :, :]
        img_low_exposure_3 = img_low[:, 9:12, :, :]
        f1 = self.conv1(img_low_raw)
        f2 = self.conv1(img_low_exposure_1)
        f3 = self.conv1(img_low_exposure_2)
        f4 = self.conv1(img_low_exposure_3)
        j1 = f2 - f1
        j2 = f3 - f2
        j3 = f4 - f3
        j3 = torch.cat([j1, j2, j3], dim=1)
        j3_conv = self.conv(j3)
        add = self.local_net(img_low_raw)
        add1 = self.local_net_1(j3_conv)
        add1 = add1.mul(self.nlt_weight) + self.nlt_bias
        img_high = add + add1
        if not self.with_global:
            mul = add
            return mul, add, img_high
        else:
            pool = nn.AvgPool2d(2, stride=2)
            img_low = pool(img_high)
            col_calib = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)
            img_high = torch.stack([self.apply_color(img_high[i, :, :, :], col_calib[i, :, :]) for i in range(b)],
                                   dim=0)
            img_high = img_high.permute(0, 3, 1, 2)
            return add, add, img_high
