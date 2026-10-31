import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=2)
        w = self.fc(w).unsqueeze(2)
        return x * w

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcitation1d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.gelu(res + h)

class FilterBankTCNEEG(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        # 4 temporal filterbank branches for delta, theta, alpha/beta, gamma
        self.fb1 = nn.Conv1d(n_channels, 16, kernel_size=63, stride=2, padding=31, bias=False)
        self.fb2 = nn.Conv1d(n_channels, 16, kernel_size=31, stride=2, padding=15, bias=False)
        self.fb3 = nn.Conv1d(n_channels, 16, kernel_size=15, stride=2, padding=7, bias=False)
        self.fb4 = nn.Conv1d(n_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_stem = nn.BatchNorm1d(64)
        
        self.spatial_mix = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn_mix = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.tcn1 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=1)
        self.tcn2 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=2)
        self.tcn3 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=4)
        self.tcn4 = DilatedResidualBlock(channels=64, kernel_size=5, dilation=8)
        
        self.attn_pool = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.norm_out = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, n_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.fb1(x)
        b2 = self.fb2(x)
        b3 = self.fb3(x)
        b4 = self.fb4(x)
        h = torch.cat([b1, b2, b3, b4], dim=1)
        h = F.gelu(self.bn_stem(h))
        h = self.pool(F.gelu(self.bn_mix(self.spatial_mix(h))))
        
        h = self.tcn1(h)
        h = self.tcn2(h)
        h = self.tcn3(h)
        h = self.tcn4(h)
        
        h = h.transpose(1, 2)
        scores = self.attn_pool(h)
        weights = F.softmax(scores, dim=1)
        pooled = (h * weights).sum(dim=1)
        pooled = self.norm_out(pooled)
        return self.classifier(pooled)

def build_model(n_channels: int, n_samples: int, n_classes: int) -> nn.Module:
    return FilterBankTCNEEG(n_channels=n_channels, n_classes=n_classes)
