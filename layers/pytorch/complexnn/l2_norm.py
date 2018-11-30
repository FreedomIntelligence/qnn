# -*- coding: utf-8 -*-

import torch

class L2Norm(torch.nn.Module):

    def __init__(self, dim=1, keep_dims=True):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.keepdim = keep_dims

    def forward(self, inputs):

        output = torch.sqrt(0.00001+ torch.sum(inputs**2, dim=self.dim, keepdim=self.keepdim))

        return output

def test():
    l2_norm = L2Norm()
    a = torch.randn(3,4)
    a_norm = l2_norm(a)
    # print(a_norm)
    if a_norm.size(1) == 1:
        print('L2Norm Test Passed.')
    else:
        print('L2Norm Test Failed.')

if __name__ == '__main__':
    test()