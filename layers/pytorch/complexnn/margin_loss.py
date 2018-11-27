# -*- coding: utf-8 -*-

import torch

class MarginLoss(torch.nn.Module):

    def __init__(self, margin=1):
        super(MarginLoss, self).__init__()
        self.margin = margin
        
    def forward(self, inputs):
        score1, score2 = inputs
        zeros = torch.zeros_like(score1)
        output = torch.max(score2 - score1 + self.margin, zeros)
        return output

def test():
    margin_loss = MarginLoss()
    a = torch.randn(3,4)
    b = torch.randn(3,4)
    loss = margin_loss([a, b])
    if loss.size(1) == a.size(1):
        print('MarginLoss Test Passed.')
    else:
        print('MarginLoss Test Failed.')

if __name__ == '__main__':
    test()