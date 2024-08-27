import torch
import torch.utils.data as Data
import glob
import random
import os
import numpy as np
import time

import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import vgg19
from torchvision.models import vgg16
import math



def adjust_learning_rate(optimizer, epoch, lrr):
    lr = lrr * (0.99 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16_model = vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16_model.features[:5])
        self.enc_2 = nn.Sequential(*vgg16_model.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16_model.features[10:17])

        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        out = results[1:]
        return out


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, output, gt):
        loss_dict = {}

        if output.shape[1] == 3:
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output = self.extractor(torch.cat([output] * 3, 1))
            feat_gt = self.extractor(torch.cat([gt] * 3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output)
        l_style = loss_dict['style']
        l_tv = loss_dict['tv']
        return l_style, l_tv
























    