import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class nf_se_conv_block(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 se_rate=0.5):
        super(nf_se_conv_block, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.hidden_channels = max(1, int(in_channels * se_rate))

        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1),
            torch.nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y