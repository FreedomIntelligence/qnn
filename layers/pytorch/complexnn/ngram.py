# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np

class NGram(torch.nn.Module):
    '''
    Input can be a sequence of indexes or a sequence of embeddings
    gram_n is the value of n
    dim is the dimension to which n-gram is applied
    out_n = seq_len+(n_gram-1)*2-n_gram +1 = 
    e.g. input_shape = (None,10) gram_n = 5 ==> output_shape = (None,10,5)
    e.g. input_shape = (None,10,3) gram_n = 5, axis = 1 ==> output_shape = (None,10,5,3)
    
    '''
    def __init__(self, gram_n=3, dim=1):
        super(NGram, self).__init__()
        self.gram_n = gram_n
        self.dim = dim
        
    def forward(self, inputs):
        
        slice_begin_index = 0
        slice_end_index = -1

        seq_len = inputs.size(self.dim)
        single_padded_len = self.gram_n - 1
        single_padded_range = torch.tensor(np.arange(single_padded_len), dtype=torch.long)
        single_padded_zeros = torch.zeros_like(torch.index_select(inputs, self.dim, single_padded_range))
        inputs = torch.cat([single_padded_zeros, inputs, single_padded_zeros], dim=self.dim)

        out_n = seq_len + self.gram_n - 1 
        list_of_ngrams = []
        
        for i in range(out_n):
            slice_begin_index = i
            slice_end_index = i + self.gram_n
            slice_index = torch.tensor(np.arange(slice_begin_index, slice_end_index), dtype=torch.long)
            l = torch.index_select(inputs, self.dim, index=slice_index)
            list_of_ngrams.append(torch.unsqueeze(l, dim=self.dim+1))
                
        ngram_mat = torch.cat(list_of_ngrams, dim=self.dim+1)
        
        return ngram_mat

def test():
    n_gram = NGram()
    a = torch.randn(4, 10, 3)
    n_gram_mat = n_gram(a)
    print(n_gram_mat)
    if n_gram_mat.dim() == a.dim() + 1:
        print('NGram Test Passed.')
    else:
        print('NGram Test Failed.')

if __name__ == '__main__':
    test()