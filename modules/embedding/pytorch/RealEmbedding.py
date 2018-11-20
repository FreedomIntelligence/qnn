# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F

class RealEmbedding(torch.nn.Module):
    def __init__(self, opt):
        super(RealEmbedding, self).__init__()
        self.opt = opt
        if self.opt.random_init:
            self.embedding = torch.nn.Embedding(self.opt.lookup_table.shape[0], 
                                self.opt.lookup_table.shape[1], 
                                max_norm=1,
                                norm_type=2,
                                _weight=torch.tensor(np.transpose(self.opt_lookup_table), dtype=torch.float))
        else:
            self.embedding = torch.nn.Embedding(self.opt.lookup_table.shape[0],
                                self.opt.lookup_table.shape[1],
                                max_norm=1,
                                norm_type=2)

        self.dropout_embedding = torch.nn.Dropout(1 - self.opt.dropout_rate_embedding)
            
    def get_embedding(self, doc, mask=None, use_weight=False):
        encoded = self.embedding(doc)
        if 1 - self.opt.dropout_rate_probs < 1e-6:
            encoded = self.dropout_embedding(encoded)
        
        return encoded