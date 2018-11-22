# -*- coding: utf-8 -*-

from layers.pytorch.complexnn import *

class Ngram():
    def __init__(self, opt):
        self.opt = opt
        self.n_value = int(self.opt.ngram_value)
        self.n_gram = NGram(n_value=self.n_value)
    
    def get_representation(self, seq_embedding_real, seq_embedding_imag):
        seq_embedding_real = self.n_gram(seq_embedding_real)
        seq_embedding_imag = self.n_gram(seq_embedding_imag)
        
        return seq_embedding_real, seq_embedding_imag