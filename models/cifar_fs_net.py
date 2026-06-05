import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        return x


class CifarFSNet(nn.Module):
    """
    针对32x32输入设计的网络
    32 -> 16 -> 8 -> 4 -> 4 (无pool)
    """

    def __init__(self, num_classes=5, hidden_dim=64):
        super().__init__()
        # 输入3通道
        self.layer1 = ConvBlock(3, hidden_dim)  # 32 -> 16
        self.layer2 = ConvBlock(hidden_dim, hidden_dim)  # 16 -> 8
        self.layer3 = ConvBlock(hidden_dim, hidden_dim)  # 8 -> 4

        # 第4层无pool，保持4x4分辨率
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # 4x4 = 16, 比Omniglot的3x3=9更大
        self.fc = nn.Linear(hidden_dim * 4 * 4, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    # 保持与OmniglotNet相同的接口，方便复用训练代码
    def get_flat_params(self):
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_flat_params(self, flat_params):
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view_as(p))
            idx += numel