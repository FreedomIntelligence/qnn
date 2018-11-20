# -*- coding: utf-8 -*-

import torch
import numpy as np

class Concatenation(torch.nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Concatenation, self).__init__()

    def forward(self, inputs):
        output = torch.cat(inputs, dim=self.dim)
        return output