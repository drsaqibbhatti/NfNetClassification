import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple
from block.ws_conv import ws_conv
from block.gamma_act_block import gamma_act_block
from block.nf_cbam_block import nf_cbam_block
from block.stochasticdepth_block import stochasticdepth_block

class nf_landmark_block(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 mid_dim,
                 out_dim,
                 stride=1,
                 alpha=0.2,
                 beta=1.0,
                 se_rate=0.07,
                 activation='silu',
                 kernel_size=3,
                 dilation_rate=1,
                 block_dropout_probability=0.25,
                 stochastic_probability=0.25):
        super(nf_landmark_block, self).__init__()

        self.stride = stride
        self.alpha = alpha
        self.beta = beta


        self.features = torch.nn.Sequential(ws_conv(in_dim,
                                                     mid_dim,
                                                     kernel_size=kernel_size,
                                                     stride=self.stride,
                                                     padding=int(dilation_rate*(kernel_size-1) // 2),
                                                     dilation=dilation_rate,
                                                     bias=True),
                                            gamma_act_block(activation=activation,
                                                            inplace=True),
                                            torch.nn.Dropout2d(p=block_dropout_probability),
                                            ws_conv(mid_dim,
                                                     out_dim,
                                                     kernel_size=kernel_size,
                                                     dilation=dilation_rate,
                                                     padding='same',
                                                     bias=True),
                                            gamma_act_block(activation=activation,
                                                            inplace=True),
                                            nf_cbam_block(in_channels=out_dim,
                                                   channels=out_dim,
                                                   se_rate=se_rate,
                                                   kernel_size=kernel_size,
                                                   dilation=dilation_rate),
                                            stochasticdepth_block(probability=stochastic_probability),
                                            torch.nn.Dropout2d(p=block_dropout_probability))

        self.down_skip_connection = ws_conv(in_channels=in_dim,
                                             out_channels=out_dim,
                                             kernel_size=1,
                                             stride=self.stride,
                                             bias=True)

    def forward(self, x):
        indentity = x
        if self.stride == 2:
            x = x * self.beta
            down = self.down_skip_connection(indentity)
            out = self.features(x)
            out = out * self.alpha
            out = out + down
            return out
        else:
            x = x * self.beta
            out = self.features(x)
            out = out * self.alpha
            out = out + indentity
            return out