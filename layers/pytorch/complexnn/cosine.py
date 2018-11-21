# -*- coding: utf-8 -*-

import torch
import numpy as np

class Cosinse(torch.nn.Module):

    def __init__(self, dropout_keep_prob=1, dim=-1, keep_dims=True):
        super(Cosinse, self).__init__()
        self.dim = dim
        self.keep_dims = keep_dims
        self.dropout_keep_prob = dropout_keep_prob
        self.dropout_probs = torch.nn.Dropout(p=1-dropout_keep_prob)

    def forward(self, inputs):

        x, y = inputs

        norm1 = torch.sqrt(0.00001 + torch.sum(x**2, dim=self.dim, keepdim=False))
        norm2 = torch.sqrt(0.00001 + torch.sum(y**2, dim=self.dim, keepdim=False))
        output= torch.sum(self.dropout_probs(x*y), dim=self.dim) / norm1 / norm2

        return torch.unsqueeze(output, dim=-1)

