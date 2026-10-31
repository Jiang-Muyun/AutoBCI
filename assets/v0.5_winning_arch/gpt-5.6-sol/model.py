"""
Define the winning multi-scale temporal convolutional architecture from the GPT-5.6 Sol search.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BroaderIntermediateStem(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.stem_short = nn.Conv1d(n_channels, 32, 11, stride=4, padding=5, bias=False)
        self.stem_long = nn.Conv1d(n_channels, 32, 47, stride=4, padding=23, bias=False)
        self.stem_norm = nn.GroupNorm(8, 64)
        self.depthwise = nn.Conv1d(64, 64, 31, padding=15, groups=64, bias=False)
        self.mix = nn.Conv1d(64, 64, 1, bias=False)
        self.mix_norm = nn.GroupNorm(8, 64)
        self.hidden = nn.Linear(64 * 30, 256)
        self.hidden_norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = torch.cat((self.stem_short(x), self.stem_long(x)), dim=1)
        x = F.gelu(self.stem_norm(x))
        residual = x
        x = self.mix(self.depthwise(x))
        x = F.gelu(self.mix_norm(x + residual))
        features = torch.cat((
            F.adaptive_avg_pool1d(x, 1).flatten(1),
            F.adaptive_avg_pool1d(x, 2).flatten(1),
            F.adaptive_avg_pool1d(x, 4).flatten(1),
            F.adaptive_avg_pool1d(x, 8).flatten(1),
            F.adaptive_max_pool1d(x, 1).flatten(1),
            F.adaptive_max_pool1d(x, 2).flatten(1),
            F.adaptive_max_pool1d(x, 4).flatten(1),
            F.adaptive_max_pool1d(x, 8).flatten(1)
        ), dim=1)
        features = F.gelu(self.hidden_norm(self.hidden(features)))
        return self.classifier(features)


def build_model(n_channels: int, n_samples: int, n_classes: int):
    return BroaderIntermediateStem(n_channels, n_classes)
