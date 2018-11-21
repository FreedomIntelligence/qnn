# -*- coding: utf-8 -*-

import torch
from layers.pytorch.complexnn import *

class Pooling():
    def __init__(self, opt):
        self.opt = opt
        self.dropout_embedding = torch.nn.Dropout(1 - self.opt.dropout_rate_embedding)

    def get_representation(self, encoded):
        if 1 - self.opt.dropout_rate_probs < 1e-6:
            encoded = self.dropout_embedding(encoded)
            
        representation = []
        for one_type in self.opt.pooling_type.split(','):
            if one_type == 'max':
                probs = torch.max(encoded, dim=1)
            elif one_type == 'average':
                probs = torch.mean(encoded, dim=1)
            elif one_type == 'none':
                probs = torch.view(encoded.size(0), -1).contiguous()
            elif one_type == 'max_col':
                probs = torch.max(torch.transpose(encoded, 1, 2), dim=1)
            elif one_type == 'average_col':
                probs = torch.mean(torch.transpose(encoded, 1, 2), dim=1)
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = torch.view(encoded.size(0), -1).contiguous()
            representation.append(probs)
        
        if len(representation) > 1:
            representation = torch.cat(representation, dim=-1)
        else:
            representation = representation[0]
        return representation