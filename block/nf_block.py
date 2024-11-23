import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.ws_conv import ws_conv
from block.gamma_act_block import gamma_act_block
from block.stochasticdepth_block import stochasticdepth_block
from block.nf_se_conv_block import nf_se_conv_block

class nf_block(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 mid_dim,
                 out_dim,
                 stride=1,
                 groups=32,
                 alpha=0.2,
                 beta=1.0,
                 activation='gelu',
                 stochastic_probability=0.25):
        super(nf_block, self).__init__()

        self.stride = stride
        self.alpha = alpha
        self.beta = beta
        self.groups = groups


        self.pre_activation = gamma_act_block(activation=activation)

        self.features = torch.nn.Sequential(ws_conv(in_dim,
                                                     mid_dim,
                                                     kernel_size=1,
                                                     bias=True),
                                            gamma_act_block(activation=activation,
                                                            inplace=True),
                                            ws_conv(mid_dim,
                                                     mid_dim,
                                                     kernel_size=3,
                                                     padding=1,
                                                     stride=self.stride,
                                                     bias=True,
                                                     groups=self.groups),
                                            gamma_act_block(activation=activation,
                                                            inplace=True),
                                            ws_conv(mid_dim,
                                                     mid_dim,
                                                     kernel_size=3,
                                                     padding=1,
                                                     stride=1,
                                                     bias=True,
                                                     groups=self.groups),
                                            gamma_act_block(activation=activation,
                                                            inplace=True),
                                            ws_conv(mid_dim,
                                                     out_dim,
                                                     kernel_size=1,
                                                     bias=True),
                                            nf_se_conv_block(in_channels=out_dim,
                                                          out_channels=out_dim),
                                            stochasticdepth_block(probability=stochastic_probability))


        self.down_skip_connection = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3,
                               stride=2,
                               padding=1),
            ws_conv(in_channels=in_dim,
                     out_channels=out_dim,
                     kernel_size=1,
                     stride=1,
                     bias=True)
        )

    def forward(self, x):
        indentity = x
        if self.stride == 2:
            x = x * self.beta
            x = self.pre_activation(x)
            down = self.down_skip_connection(indentity)
            out = self.features(x)
            out = out * self.alpha
            out = out + down
            return out

        else:
            x = x * self.beta
            x = self.pre_activation(x)
            out = self.features(x)
            out = out * self.alpha
            out = out + indentity
            return out