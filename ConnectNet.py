import torch
from torch import nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
       # self.conv1 = nn.Conv2d(3, 1024, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn1 = nn.BatchNorm2d(1024).cuda()
        #self.conv2 = nn.Conv2d(1024, 2048, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn2 = nn.BatchNorm2d(2048).cuda()
        #self.conv3 = nn.Conv2d(2048, 4096, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn3 = nn.BatchNorm2d(4096).cuda()
        #self.conv4 = nn.Conv2d(4096, 2048, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn4 = nn.BatchNorm2d(2048).cuda()
        #self.conv5 = nn.Conv2d(2048, 512, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn5 = nn.BatchNorm2d(512).cuda()
        self.conv1 = nn.Conv2d(26, 1024 , kernel_size=12, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 35, kernel_size=12, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(35)

        #self.conv3 = nn.Conv2d(2048, 4096, kernel_size=4, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm2d(4096)
        #self.conv4 = nn.Conv2d(4096, 2048, kernel_size=4, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(2048)
        #self.conv5 = nn.Conv2d(2048, 512, kernel_size=4, stride=1, padding=1)
        #self.bn5 = nn.BatchNorm2d(512)


    def forward(self, s):
        s = s.view(-1,26, 54, 54)  # batch_size x channels x board_x x board_y
        s = F.leaky_relu(self.bn1(self.conv1(s)))
        s = F.leaky_relu(self.bn2(self.conv2(s)))
        #s = F.relu(self.bn3(self.conv3(s)))
        #s = F.relu(self.bn4(self.conv4(s)))
        #s = F.relu(self.bn5(self.conv5(s)))

        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=35  , planes=35, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False).cuda()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes).cuda()
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False).cuda()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes).cuda()
        #self.drp = nn.Dropout(0.3).cuda()
        self.bn2 = nn.BatchNorm2d(planes)
        self.drp = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.leaky_relu(self.bn1(out))
        out = self.drp(out)
        out = self.conv2(out)
        out = F.leaky_relu(self.bn2(out))
        out = self.drp(out)
        out += residual
        out = F.leaky_relu(out)
        return out


class OutBlock(nn.Module):
    # shape=6*7*32
    shape1 = 45360
    # shape = 24156*25*25
    shape = 23436

    def __init__(self):
        super(OutBlock, self).__init__()


        self.fc1 = nn.Linear(self.shape1,1024)
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
        #self.conv = ConvBlock().cuda()
        self.conv = ConvBlock()
        for block in range(30):
            #setattr(self, "res_%i" % block, ResBlock().cuda())
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(30):
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
        # if isinstance(self.conv.conv1, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv1.weight)
        #     torch.nn.init.zeros_(self.conv.conv1.bias)
        # if isinstance(self.conv.bn1, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn1.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn1.bias.data, 0)
        # if isinstance(self.conv.conv2, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv2.weight)
        #     torch.nn.init.zeros_(self.conv.conv2.bias)
        # if isinstance(self.conv.bn2, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn2.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn2.bias.data, 0)
        # if isinstance(self.conv.conv3, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv3.weight)
        #     torch.nn.init.zeros_(self.conv.conv3.bias)
        # if isinstance(self.conv.bn3, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn3.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn3.bias.data, 0)
        # if isinstance(self.conv.conv4, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv4.weight)
        #     torch.nn.init.zeros_(self.conv.conv4.bias)
        # if isinstance(self.conv.bn4, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn4.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn4.bias.data, 0)
        # if isinstance(self.conv.conv5, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv5.weight)
        #     torch.nn.init.zeros_(self.conv.conv5.bias)
        # if isinstance(self.conv.bn5, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn5.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn5.bias.data, 0)
