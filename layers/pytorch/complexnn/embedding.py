# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np

def PhaseEmbedding(input_dim, embedding_dim):
    embedding_layer = torch.nn.Embedding(input_dim, embedding_dim, padding_idx=0)
    torch.nn.init.uniform_(embedding_layer.weight, 0, 2*np.pi)
    return embedding_layer

def AmplitudeEmbedding(embedding_matrix, random_init=True):
    embedding_dim = embedding_matrix.shape[1]
    vocabulary_size = embedding_matrix.shape[0]
    if random_init:
        # Normal(0, 1)
        return torch.nn.Embedding(vocabulary_size,
                        embedding_dim,
#                        max_norm=1,
#                        norm_type=2,
                        padding_idx=0)
    else:
        return torch.nn.Embedding(vocabulary_size,
                        embedding_dim,
#                        max_norm=1,
#                        norm_type=2,
                        padding_idx=0,
                        _weight=torch.tensor(embedding_matrix, dtype=torch.float))

class ComplexEmbedding(torch.nn.Module):
    def __init__(self, embedding_matrix, sentiment_lexicon=None, freeze=False):
        super(ComplexEmbedding, self).__init__()
        if sentiment_lexicon is None:
            sign_matrix = torch.sign(embedding_matrix)
            amplitude_embedding_matrix = sign_matrix * embedding_matrix
            self.amplitude_embed = nn.Embedding.from_pretrained(amplitude_embedding_matrix, freeze=freeze)
            phase_embedding_matrix = math.pi * (1 - sign_matrix) / 2 # based on [0, 2*pi]
            self.phase_embed = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=freeze)
        else:
            self.positive_indices = sentiment_lexicon.gt(0.).squeeze(-1).float()
            self.negative_indices = sentiment_lexicon.lt(0.).squeeze(-1).float()
            self.sampler = torch.Tensor(sentiment_lexicon.size(0))
            sentiment_lexicon = 2 * torch.acos(sentiment_lexicon).expand(-1, 50)
            self.amplitude_embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze)
            self.phase_embed = nn.Embedding.from_pretrained(sentiment_lexicon, freeze=freeze)

    def sample(self, num):
        self.sampler.uniform_(0, 1)
        positive_prior = self.positive_indices * self.sampler
        negative_prior = self.negative_indices * self.sampler
        positive_samples = positive_prior.multinomial(num)
        negative_samples = negative_prior.multinomial(num)
        samples = torch.cat([positive_samples, negative_samples])
        return self.forward(samples)

    def forward(self, indices):
        amplitude_embed = self.amplitude_embed(indices)
        phase_embed = self.phase_embed(indices)

        return [amplitude_embed, phase_embed]
    

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