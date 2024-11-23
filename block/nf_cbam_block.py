import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.nf_se_conv_block import nf_se_conv_block
from block.nf_spatial_block import nf_spatial_block

class nf_cbam_block(torch.nn.Module):
    def __init__(self, in_channels, channels, se_rate=0.5, kernel_size=7, dilation=1):
        super(nf_cbam_block, self).__init__()
        #Squeeze-and-excitiation
        self.channel_wise_conv = nf_se_conv_block(in_channels=in_channels,
                                               out_channels=channels,
                                               se_rate=se_rate)
        self.spatial_wise_conv = nf_spatial_block(kernel_size=kernel_size,
                                               dilation=dilation)
    def forward(self, x):
        x_out = self.channel_wise_conv(x)
        x_out = self.spatial_wise_conv(x_out)
        return x_out