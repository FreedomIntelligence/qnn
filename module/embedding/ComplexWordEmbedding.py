# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm

from layers.keras.complexnn import *

from keras.initializers import Constant
import numpy as np
import math

class ComplexWordEmbedding(BasicModel):
    
    def initialize(self):
       
#        self.phase_embedding=phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable)
#                
#        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init)
#
#        
        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), None, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
        self.phase_embedding= phase_embedding_layer(None, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)
        
        self.l2_normalization = L2Normalization(axis = 3)
        self.l2_norm = L2Norm(axis = 3, keep_dims = False)
        
    def __init__(self,opt):
        super(ComplexWordEmbedding, self).__init__(opt) 
    
    def get_amplitude_encoded(self,doc,mask=None):
        return self.amplitude_embedding(doc)
        
    def process_complex_embedding(self,amplitude_encoded,use_weight=False):
        phase_encoded = self.phase_embedding(doc)
        if use_weight:
            self.weight = Activation('softmax')(self.l2_norm(amplitude_encoded))
#            print(self.weight.shape)
#            self.weight = reshape((-1,self.opt.max_sequence_length,self.opt.ngram_value,1))(self.weight)
            self.amplitude_encoded = self.l2_normalization(amplitude_encoded)  
        else:
            self.weight = None
        if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
            self.phase_encoded = self.dropout(phase_encoded)
            self.amplitude_encoded = self.dropout(amplitude_encoded)
            
        [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([ phase_encoded, amplitude_encoded])
            
        return seq_embedding_real,seq_embedding_imag,self.weight
        
    def get_embedding(self,doc,use_weight=False):

        amplitude_encoded = self.get_amplitude_encoded(doc)
        return process_complex_embedding(amplitude_encoded,use_weight=use_weight)

