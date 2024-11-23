import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple


class stochasticdepth_block(torch.nn.Module):
    def __init__(self,
                 probability):
        super().__init__()

        self.probability = 1 - probability

    def forward(self, x):
        if self.training:
            pmask = torch.bernoulli(torch.tensor(self.probability))
            x = x * pmask
            #x[:, :, :, :] = pmask
            return x
        else:
            return x