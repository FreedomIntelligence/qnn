# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class L2Normalization(torch.nn.Module):

    def __init__(self, p=2, dim=1, eps=1e-12):
        super(L2Normalization, self).__init__()
        self.dim = dim
        self.p = p
        self.eps = eps

    def forward(self, inputs):
        # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
        # v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
        output = F.normalize(inputs, p=self.p, dim=self.dim, eps=self.eps)
        return output

def test():
    l2_normalization = L2Normalization()
    a = torch.randn(3,4)
    a_normalization = l2_normalization(a)
    print(a)
    print(a_normalization)
    if a_normalization.size() == a.size():
        print('L2Normalization Test Passed.')
    else:
        print('L2Normalization Test Failed.')

if __name__ == '__main__':
    test()