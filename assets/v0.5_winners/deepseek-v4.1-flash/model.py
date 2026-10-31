import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadKernelCovPool(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=48, k_long=51, k_mid=25, k_short=13, k_fine=7,
                 mix_dim=128, drop=0.3):
        super().__init__()
        self.stem_long = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, k_long, stride=2, padding=k_long // 2, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
        )
        self.stem_mid = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, k_mid, stride=2, padding=k_mid // 2, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
        )
        self.stem_short = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, k_short, stride=2, padding=k_short // 2, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
        )
        self.stem_fine = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, k_fine, stride=2, padding=k_fine // 2, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.GELU(),
        )
        self.reduce = nn.Sequential(
            nn.Conv1d(n_channels * 4, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        idx = torch.triu_indices(hidden, hidden)
        self.register_buffer('tri_i', idx[0].contiguous(), persistent=False)
        self.register_buffer('tri_j', idx[1].contiguous(), persistent=False)
        feat = hidden * (hidden + 1) // 2
        self.norm = nn.LayerNorm(feat)
        self.proj = nn.Linear(feat, mix_dim)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(mix_dim, n_classes)

    def forward(self, x):
        a = self.stem_long(x)
        b = self.stem_mid(x)
        c = self.stem_short(x)
        d = self.stem_fine(x)
        t = min(a.shape[-1], b.shape[-1], c.shape[-1], d.shape[-1])
        h = torch.cat([a[..., :t], b[..., :t], c[..., :t], d[..., :t]], dim=1)
        h = self.reduce(h)
        h = h - h.mean(dim=-1, keepdim=True)
        hf = h.float()
        cov = torch.matmul(hf, hf.transpose(1, 2)) / float(max(1, hf.shape[-1]))
        z = cov[:, self.tri_i, self.tri_j]
        z = z.to(self.norm.weight.dtype)
        z = self.norm(z)
        z = self.proj(z)
        z = F.gelu(z)
        z = self.drop(z)
        return self.head(z)


def build_model(n_channels: int, n_samples: int, n_classes: int):
    return QuadKernelCovPool(n_channels, n_classes)
