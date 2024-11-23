import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.ws_conv import ws_conv
from block.channelpool_block import channelpool_block

class nf_spatial_block(torch.nn.Module):
    def __init__(self, kernel_size=7, dilation=1):
        super(nf_spatial_block, self).__init__()
        self.compress = channelpool_block()
        self.spatial = torch.nn.Sequential(
            ws_conv(in_channels=2,
                     out_channels=1,
                     kernel_size=kernel_size,
                     stride=1,
                     padding=int(dilation * (kernel_size-1) // 2),
                     bias=True,
                     dilation=dilation),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale