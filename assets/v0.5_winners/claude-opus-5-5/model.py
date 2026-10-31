import torch
import torch.nn as nn
import torch.nn.functional as F


def _standardize(x):
    xf = x.float()
    xf = xf - xf.mean(dim=2, keepdim=True)
    s = xf.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
    return (xf / (s + 1e-6)).to(x.dtype)


class InceptionBlock(nn.Module):
    def __init__(self, cin, cb, ks):
        super().__init__()
        cout = cb * (len(ks) + 1)
        self.bottleneck = nn.Conv1d(cin, cb, 1, bias=False)
        self.branches = nn.ModuleList([nn.Conv1d(cb, cb, k, padding=k // 2, bias=False) for k in ks])
        self.pool_branch = nn.Conv1d(cin, cb, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        hid = max(cout // 4, 8)
        self.se = nn.Sequential(nn.Linear(cout, hid), nn.ReLU(), nn.Linear(hid, cout), nn.Sigmoid())

    def forward(self, x):
        b = self.bottleneck(x)
        outs = [br(b) for br in self.branches]
        outs.append(self.pool_branch(F.max_pool1d(x, 3, stride=1, padding=1)))
        y = F.elu(self.bn(torch.cat(outs, dim=1)))
        return y * self.se(y.mean(dim=-1)).unsqueeze(-1)


class Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.spatial = nn.Conv1d(n_channels, 32, 7, padding=3, bias=False)
        self.bn0 = nn.BatchNorm1d(32)
        self.block1 = InceptionBlock(32, 16, (9, 31, 95))
        self.block2 = InceptionBlock(64, 24, (5, 15, 45))
        self.block3 = InceptionBlock(96, 32, (3, 9, 27))
        self.block4 = InceptionBlock(128, 32, (3, 7, 15))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.bn0(self.spatial(_standardize(x)))
        x = self.block1(x)
        x = F.avg_pool1d(x, 4)
        x = self.block2(x)
        x = F.avg_pool1d(x, 4)
        x = self.block3(x)
        x = F.avg_pool1d(x, 2)
        x = self.block4(x)
        m = x.mean(dim=-1)
        s = x.float().var(dim=-1, unbiased=False).add(1e-5).sqrt().to(x.dtype)
        return self.fc(self.drop(torch.cat([m, s], dim=1)))


def build_model(n_channels: int, n_samples: int, n_classes: int):
    return Net(n_channels, n_classes)
