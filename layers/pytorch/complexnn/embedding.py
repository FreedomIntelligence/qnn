# -*- coding: utf-8 -*-

import torch
import numpy as np

def PhaseEmbedding(input_dim, embedding_dim):
    embedding_layer = torch.nn.Embedding(input_dim, embedding_dim, padding_idx=0)
    torch.nn.init.uniform_(embedding_layer.weight, 0, 2*np.pi)
    return embedding_layer

def AmplitudeEmbedding(embedding_matrix, random_init=True):
    embedding_dim = embedding_matrix.shape[0]
    vocabulary_size = embedding_matrix.shape[1]
    if random_init:
        # Normal(0, 1)
        return torch.nn.Embedding(vocabulary_size,
                        embedding_dim,
                        max_norm=1,
                        norm_type=2,
                        padding_idx=0)
    else:
        return torch.nn.Embedding(vocabulary_size,
                        embedding_dim,
                        max_norm=1,
                        norm_type=2,
                        padding_idx=0,
                        _weight=torch.tensor(np.transpose(embedding_matrix), dtype=torch.float))

def test():
    phase_embed = PhaseEmbedding(5, 10)
    mat = np.random.randn(10, 5)
    amplitude_embed = AmplitudeEmbedding(mat, False)
    a = torch.tensor([[0,1,2]], dtype=torch.long)
    b = torch.tensor([[0,1,2]], dtype=torch.long)
    a_out = phase_embed(a)
    b_out = amplitude_embed(b)
    if a_out.dim() == 3 and b_out.dim() == 3:
        print('Embedding Test Passed.')
    else:
        print('Embedding Test Failed.')

if __name__ == '__main__':
    test()