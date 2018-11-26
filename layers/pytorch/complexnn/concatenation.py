# -*- coding: utf-8 -*-

import torch

class Concatenation(torch.nn.Module):
    def __init__(self, dim=-1):
        super(Concatenation, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        output = torch.cat(inputs, dim=self.dim)
        return output

def test():
    a = torch.randn(3,4)
    b = torch.randn(3,2)
    concat = Concatenation(-1)
    c = concat([a, b])
    if c.size(-1) == 6:
        print('Concantenation Test Passed.')
    else:
        print('Concantenation Test Failed.')

if __name__ == '__main__':
    test()