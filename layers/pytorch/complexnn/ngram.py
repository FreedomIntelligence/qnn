# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import math
import numpy as np

class NGram(torch.nn.Module):
    '''
    Input can be a sequence of indexes or a sequence of embeddings
    n_value is the value of n
    dim is the dimension to which n-gram is applied
    
    e.g. input_shape = (None,10) n_value = 5 ==> output_shape = (None,10,5)
    
    e.g. input_shape = (None,10,3) n_value = 5, axis = 1 ==> output_shape = (None,10,5,3)
    
    '''
    def __init__(self, n_value=3, dim=1):
        super(NGram, self).__init__()
        self.n_value = n_value
        self.dim = dim
        
    def forward(self, inputs):
        
        slice_begin_index = 0
        slice_end_index = -1
        seq_len = inputs.size(self.dim)
        list_of_ngrams = []
        
        for i in range(self.n_value):
            begin = max(0,i-math.floor(self.n_value/2))
            end = min(seq_len-1+i-math.floor(self.n_value/2),seq_len-1)
            slice_begin_index = begin
            slice_end_index = end-begin+1
            slice_index = torch.tensor(np.arange(slice_begin_index, slice_end_index), dtype=torch.long)
            l = torch.index_select(inputs, self.dim, index=slice_index)
            
            slice_begin_index = 0
            slice_end_index = int(seq_len-(end-begin+1))
#            print(slice_end_index)
            slice_index = torch.tensor(np.arange(slice_begin_index, slice_end_index), dtype=torch.long)
            padded_zeros = torch.zeros_like(torch.index_select(inputs, self.dim, slice_index))
#            print(padded_zeros.shape)
            if begin == 0:
                #left_padding
                list_of_ngrams.append(torch.unsqueeze(torch.cat([padded_zeros,l], dim=self.dim), dim=self.dim+1))
                #right_padding
            else:
                list_of_ngrams.append(torch.unsqueeze(torch.cat([l,padded_zeros], dim=self.dim), dim=self.dim+1))
                
        ngram_mat = torch.cat(list_of_ngrams, dim=self.dim+1)
        
        return ngram_mat