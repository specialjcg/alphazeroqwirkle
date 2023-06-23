import torch
import torch.nn as nn
import torch.nn.functional as F

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


conv_block = ConvBlock()
input_tensor = torch.randn(1, 26, 54, 54)  # Example input with batch_size=1
output_tensor = conv_block(input_tensor)
output_tensor = output_tensor.view(1, -1)
print(output_tensor.shape)
print(output_tensor.shape)
