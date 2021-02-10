import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=126, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=126)

    def forward(self, x):
        x = x.view(-1, 3, 6, 7)
        x = F.relu(self.bn1(self.conv1(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=126, out_channels=126, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=126)
        self.conv2 = nn.Conv2d(in_channels=126, out_channels=126, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=126)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=126, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.fc1 = nn.Linear(in_features=1*6*7, out_features=3*6*7, bias=True)
        self.fc2 = nn.Linear(in_features=3*6*7, out_features=1, bias=True)

    def forward(self, x):
        v = F.relu(self.bn1(self.conv1(x)))
        v = v.view(-1, 6*7)
        v = F.relu(self.fc1(v))
        v = self.fc2(v)
        v = torch.tanh(v)
        return v


class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=126, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=2)
        self.fc1 = nn.Linear(in_features=2*6*7, out_features=7)

    def forward(self, x):
        p = F.relu(self.bn1(self.conv1(x)))
        p = p.view(-1, 2*6*7)
        p = F.log_softmax(self.fc1(p), dim=1).exp()
        return p


class AlphaNet(nn.Module):
    def __init__(self):
        super(AlphaNet, self).__init__()
        self.conv = ConvBlock()
        for i in range(10):
            setattr(self, f'res{i}', ResidualBlock())
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, x):
        x = self.conv(x)
        for i in range(10):
            x = getattr(self, f'res{i}')(x)
        v = self.value_head(x)
        p = self.policy_head(x)
        return v, p
