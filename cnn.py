import torch
from torch import nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(26, 64, kernel_size=6, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=6, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 240, kernel_size=6, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(240)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)

    def forward(self, s):
        s = s.view(-1, 26, 54, 54)
        s = F.leaky_relu(self.bn1(self.conv1(s)))
        s = self.pool1(s)
        s = self.dropout1(s)
        s = F.leaky_relu(self.bn2(self.conv2(s)))
        s = self.pool2(s)
        s = self.dropout2(s)
        s = F.leaky_relu(self.bn3(self.conv3(s)))
        s = self.dropout3(s)
        s = F.leaky_relu(self.bn4(self.conv4(s)))
        s = self.dropout4(s)
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=240, planes=240, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes // 4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // 4)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // 4)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.drp = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.leaky_relu(self.bn1(out))
        out = self.drp(out)

        out = self.conv2(out)
        out = F.leaky_relu(self.bn2(out))
        out = self.drp(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.leaky_relu(out)
        out = self.drp(out)

        return out


class OutBlock(nn.Module):
    # shape=6*7*32
    shape1 = 24000
    # shape = 24156*25*25
    shape = 23436

    def __init__(self):
        super(OutBlock, self).__init__()

        self.fc1 = nn.Linear(self.shape1, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.drp = nn.Dropout(0.3)

        self.fc = nn.Linear(self.shape1, self.shape1)
        self.fcinter = nn.Linear(self.shape1, self.shape)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, s):
        v = s.view(-1, self.shape1)  # batch_size X channel X height X width
        v = self.drp(F.leaky_relu(self.fc1(v)))
        v = torch.tanh(self.fc2(v))

        p = s.view(-1, self.shape1)
        p = self.drp(F.leaky_relu(self.fc(p)))
        p = self.fcinter(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        # self.conv = ConvBlock().cuda()
        self.conv = ConvBlock()
        for block in range(20):
            # setattr(self, "res_%i" % block, ResBlock().cuda())
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(20):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

    def init_weights(self):
        """
        Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
        "nn.Module"
        :param m: Layer to initialize
        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
