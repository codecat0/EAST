"""
@File : east.py
@Author : CodeCat
@Time : 2021/7/12 下午3:49
"""
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
import math

from model.resnet import ResNet, Bottleneck, model_urls


class ConvBNReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBNReLu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Extractor(nn.Module):
    def __init__(self, pretrained):
        super(Extractor, self).__init__()
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            resnet50.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.resnet50 = resnet50

    def forward(self, x):
        _, f = self.resnet50(x)
        return f


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()

        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cbr1 = ConvBNReLu(3072, 128, 1)
        self.cbr2 = ConvBNReLu(128, 128, 3)

        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cbr3 = ConvBNReLu(640, 64, 1)
        self.cbr4 = ConvBNReLu(64, 64, 3)

        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cbr5 = ConvBNReLu(320, 64, 1)
        self.cbr6 = ConvBNReLu(64, 32, 3)

        self.cbr7 = ConvBNReLu(32, 32, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x[3] -> [batch_size, 2048, w/32, h/32]
        # x[2] -> [batch_size, 1024, w/16, h/16]
        # x[1] -> [batch_size, 512, w/8, h/8]
        # x[0] -> [batch_size, 256, w/4, h/4]
        y = self.unpool1(x[3])      # [batch_size, 2048, w/16, h/16]
        y = torch.cat((y, x[2]), dim=1)     # [batch_size, 3072, w/16, h/16]
        y = self.cbr1(y)
        y = self.cbr2(y)    # [batch_size, 128, w/16, h/16]

        y = self.unpool2(y)     # [batch_size, 128, w/8, h/8]
        y = torch.cat((y, x[1]), dim=1)     # [batch_size, 640, w/8, h/8]
        y = self.cbr3(y)
        y = self.cbr4(y)    # [batch_size, 64, w/8, h/8]

        y = self.unpool3(y)     # [batch_size, 64, w/4, h/4]
        y = torch.cat((y, x[0]), dim=1)     # [batch_size, 320, w/4, h/4]
        y = self.cbr5(y)
        y = self.cbr6(y)    # [batch_size, 32, w/4, h/4]

        y = self.cbr7(y)    # [batch_size, 32, w/4, h/4]
        return y


class Output(nn.Module):
    def __init__(self, scope=512):
        super(Output, self).__init__()
        # score map
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        # RBOX geometry
        # text boxes
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        # text rotation angle
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x -> [batch_size, 32, w/4, h/4]
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi / 2
        geo = torch.cat((loc, angle), dim=1)
        return score, geo


class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        self.extractor = Extractor(pretrained)
        self.merge = Merge()
        self.output = Output()

    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))
