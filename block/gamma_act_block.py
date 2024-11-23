import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


#NFNet Normalization Free Module
_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)


_nonlin_table = dict(
    identity=torch.nn.Identity,
    celu=torch.nn.CELU,
    elu=torch.nn.ELU,
    gelu=torch.nn.GELU,
    leaky_relu=torch.nn.LeakyReLU,
    log_sigmoid=torch.nn.LogSigmoid,
    log_softmax=torch.nn.LogSoftmax,
    relu=torch.nn.ReLU,
    relu6=torch.nn.ReLU6,
    selu=torch.nn.SELU,
    sigmoid=torch.nn.Sigmoid,
    silu=torch.nn.SiLU,
    softsign=torch.nn.Softsign,
    softplus=torch.nn.Softplus,
    tanh=torch.nn.Tanh,
)

class gamma_act_block(torch.nn.Module):
    def __init__(self,
                 activation='relu',
                 inplace=False):
        super(gamma_act_block, self).__init__()

        if activation == 'gelu':
            self.activation = _nonlin_table[activation]()
        else:
            self.activation = _nonlin_table[activation](inplace=inplace)
        self.gamma = _nonlin_gamma[activation]

    def forward(self, x):
        x = self.activation(x) * self.gamma
        return x