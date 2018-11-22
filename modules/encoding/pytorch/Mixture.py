# -*- coding: utf-8 -*-

import torch
from layers.pytorch.complexnn import *

class Mixture():
    def __init__(self, opt, weights=None):
        self.opt = opt 
        self.weights = weights
        if self.weights is None:
            self.mixture = ComplexMixture(average_weights=True)
        
    
    def get_representation(self,seq_embedding_real,seq_embedding_imag,need_flatten=True):
        # If weights are passed to the mixture model, then conducting weighted sum
        # Else use uniform weights to compose the mixed state
        # Need_flatten = True if one needs to get the Flattened complex density matrix
        
        if self.weights is None:
            [sentence_embedding_real, sentence_embedding_imag]= self.mixture([seq_embedding_real, seq_embedding_imag])
        else:
            [sentence_embedding_real, sentence_embedding_imag]= self.mixture([seq_embedding_real, seq_embedding_imag, self.weights])
       
        if need_flatten:
            sentence_embedding_real = sentence_embedding_real.view(sentence_embedding_real.size(0), -1)
            sentence_embedding_imag = sentence_embedding_imag.view(sentence_embedding_imag.size(0), -1)     
             
        return [sentence_embedding_real, sentence_embedding_imag]
