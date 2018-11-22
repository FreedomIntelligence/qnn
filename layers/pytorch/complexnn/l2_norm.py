# -*- coding: utf-8 -*-

import torch

class L2Norm(torch.nn.Module):

    def __init__(self, dim=1, keep_dims=True):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.keep_dims = keep_dims

    def forward(self, inputs):

        output = torch.sqrt(0.00001+ torch.sum(inputs**2, dim=self.dim, keepdim=self.keep_dims))

        return output
