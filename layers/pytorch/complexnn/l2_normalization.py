# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class L2Normalization(torch.nn.Module):

    def __init__(self, p=2, dim=1, eps=1e-12):
        self.dim = dim
        self.p = p
        self.eps = eps
        super(L2Normalization, self).__init__()

    def forward(self, inputs):
        # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
        # v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
        output = F.normalize(inputs, p=self.p, dim=self.dim, eps=self.eps)
        return output