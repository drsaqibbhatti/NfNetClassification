import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.ws_conv import ws_conv
from block.gamma_act_block import gamma_act_block

class nf_separableConv_block(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 stride=1,
                 activation='relu6',
                 bias=False):
        super(nf_separableConv_block, self).__init__()


        self.depthwise = ws_conv(in_channels,
                                  in_channels,
                                  kernel_size=kernel_size,
                                  groups=in_channels,
                                  bias=bias,
                                  stride=stride,
                                  padding=padding)

        self.pointwise = ws_conv(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  bias=bias)


        self.activation1 = gamma_act_block(activation=activation,
                                           inplace=True)
        self.activation2 = gamma_act_block(activation=activation,
                                           inplace=True)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.activation1(out)
        out = self.pointwise(out)
        out = self.activation2(out)
        return out