# -*- coding: utf-8 -*-

import torch

class Cosine(torch.nn.Module):

    def __init__(self, dropout_keep_prob=1, dim=-1):
        super(Cosine, self).__init__()
        self.dim = dim
        self.dropout_keep_prob = dropout_keep_prob
        self.dropout = torch.nn.Dropout(p=1-dropout_keep_prob)

    def forward(self, inputs):

        x, y = inputs

        norm1 = torch.sqrt(0.00001 + torch.sum(x**2, dim=self.dim))
        norm2 = torch.sqrt(0.00001 + torch.sum(y**2, dim=self.dim))
        output = torch.sum(self.dropout(x*y), dim=self.dim) / norm1 / norm2

        return output

def test():
    cos = Cosine()
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    sim = cos([a, b])
    if sim.size(0) == 3:
        print('Cosine Test Passed.')
    else:
        print('Cosine Test Failed.')

if __name__ == '__main__':
    test()