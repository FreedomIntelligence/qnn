# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
from layers.pytorch.complexnn import *

class ComplexEmbedding(torch.nn.Module):   
    def __init__(self, opt):
        super(ComplexEmbedding, self).__init__() 
        self.amplitude_embedding = AmplitudeEmbedding(np.transpose(self.opt.lookup_table), random_init=self.opt.random_init)
        self.phase_embedding= PhaseEmbedding(self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1])
        
        self.l2_normalization = L2Normalization(dim=2)
        self.l2_norm = L2Norm(dim=2, keep_dims=False)
        self.dropout_embedding = torch.nn.Dropout(1-self.opt.dropout_rate_embedding)
        
    def process_complex_embedding(self,doc,amplitude_encoded,use_weight=False):
        phase_encoded = self.phase_embedding(doc)
        if use_weight:
            self.weight = F.softmax(self.l2_norm(amplitude_encoded))
            self.amplitude_encoded = self.l2_normalization(amplitude_encoded)  
        else:
            self.weight = None
            
        if 1 - self.opt.dropout_rate_probs < 1e-6:
            self.phase_encoded = self.dropout_embedding(phase_encoded)
            self.amplitude_encoded = self.dropout_embedding(amplitude_encoded)
            
        [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_encoded, amplitude_encoded])
            
        return seq_embedding_real,seq_embedding_imag,self.weight
        
    def get_embedding(self,doc,mask=None,use_weight=False):

        amplitude_encoded = self.amplitude_embedding(doc)
        return self.process_complex_embedding(doc,amplitude_encoded,use_weight=use_weight)