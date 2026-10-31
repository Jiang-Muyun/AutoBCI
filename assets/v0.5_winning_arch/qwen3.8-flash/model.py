import torch
import torch.nn as nn
import torch.nn.functional as F
class TemporalSmoothingPowerDispersionNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()
        self.hidden = 128
        self.channel_mix = nn.Conv1d(n_channels, self.hidden, kernel_size=1)
        self.blocks = nn.ModuleList()
        for dilation in [1, 3]:
            kernel_size = 7
            padding = (kernel_size - 1) * dilation // 2
            depthwise = nn.Conv1d(self.hidden, self.hidden, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=self.hidden)
            pointwise = nn.Conv1d(self.hidden, self.hidden, kernel_size=1)
            norm = nn.GroupNorm(8, self.hidden)
            self.blocks.append(nn.ModuleList([depthwise, pointwise, norm]))
        self.layernorm = nn.LayerNorm(self.hidden * 2)
        self.fc = nn.Linear(self.hidden * 2, n_classes)
    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=1)
        input_dtype = x.dtype
        x = F.relu(self.channel_mix(x))
        for block in self.blocks:
            depthwise, pointwise, norm = block
            residual = x
            y = F.relu(depthwise(x))
            y = pointwise(y)
            y = norm(y).to(input_dtype)
            x = F.relu(residual + y)
        x32 = x.to(torch.float32)
        mean = x32.mean(dim=-1)
        centered = x32 - mean.unsqueeze(-1)
        var = (centered * centered).mean(dim=-1)
        std = torch.sqrt(var + 1e-6).to(input_dtype)
        stats = torch.cat([mean.to(input_dtype), std], dim=1)
        input_dtype = stats.dtype
        x = self.layernorm(stats).to(input_dtype)
        x = F.relu(x)
        return self.fc(x)
def build_model(n_channels: int, n_samples: int, n_classes: int):
    return TemporalSmoothingPowerDispersionNet(n_channels, n_samples, n_classes)
