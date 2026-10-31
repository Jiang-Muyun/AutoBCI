import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        b = out_ch // 4
        rem = out_ch - 3 * b
        self.b1 = nn.Conv1d(in_ch, b, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_ch, b, kernel_size=7, padding=3)
        self.b3 = nn.Conv1d(in_ch, b, kernel_size=15, padding=7)
        self.b4 = nn.Conv1d(in_ch, rem, kernel_size=31, padding=15)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return F.elu(self.bn(y))


class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        red = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, red)
        self.fc2 = nn.Linear(red, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class InceptionTemporalSE(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=48):
        super().__init__()
        self.spatial = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.block1 = InceptionBlock(hidden, hidden)
        self.se1 = SEBlock1d(hidden, reduction=8)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = InceptionBlock(hidden, hidden * 2)
        self.se2 = SEBlock1d(hidden * 2, reduction=8)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = self.spatial(x)
        x = self.block1(x)
        x = self.se1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.pool2(x).squeeze(-1)
        return self.classifier(x)


def build_model(n_channels: int, n_samples: int, n_classes: int):
    return InceptionTemporalSE(n_channels, n_classes)
