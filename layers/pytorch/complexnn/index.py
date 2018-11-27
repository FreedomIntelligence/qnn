# -*- coding: utf-8 -*-

import torch

class Index(torch.nn.Module):

    def __init__(self, index=0, keepdim=True):
        super(Index, self).__init__()
        self.index = index
        self.keepdim = keepdim

    def forward(self, inputs):
        if self.keepdim:
            output = inputs[:, self.index, :].unsqueeze(dim=1)
        else: 
            output = inputs[:, self.index, :]
        return output

def test():
    ind = Index(index=2)
    a = torch.randn(3,4,5)
    a_index = ind(a)
    # print(a_index.size())
    if a_index.size(1) == 1:
        print('Index Test Passed.')
    else:
        print('Index Test Failed.')

if __name__ == '__main__':
    test()