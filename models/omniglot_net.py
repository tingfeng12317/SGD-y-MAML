# models/omniglot_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Omniglot标准卷积块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class OmniglotNet(nn.Module):
    """4层卷积特征网络"""

    def __init__(self, num_classes=5, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.layer1 = ConvBlock(1, hidden_dim)  # 28 -> 14
        self.layer2 = ConvBlock(hidden_dim, hidden_dim)  # 14 -> 7
        self.layer3 = ConvBlock(hidden_dim, hidden_dim)  # 7 -> 3

        # 第4层无pool
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim * 3 * 3, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_flat_params(self):
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_flat_params(self, flat_params):
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view_as(p))
            idx += numel