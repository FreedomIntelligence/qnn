# -*- coding: utf-8 -*-

import torch

class MarginLoss(torch.nn.Module):

    def __init__(self, margin=1):
        super(MarginLoss, self).__init__()
        self.margin = margin
        
    def forward(self, inputs):
        score1, score2 = inputs
        zeros = torch.zeros(score1.size())
        output = torch.max(score2 - score1 + self.margin, 0)
        return output
