# -*- coding: utf-8 -*-

import torch

class Index(torch.nn.Module):

    def __init__(self, index=0):
        super(Index, self).__init__()
        self.index = index

    def forward(self, inputs):
        output = inputs[:,self.index,:]
        return(output)
