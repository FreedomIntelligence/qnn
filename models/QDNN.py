# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from .BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
import sys
sys.path.append('complexnn')
from embedding import phase_embedding_layer, amplitude_embedding_layer
from multiply import ComplexMultiply
from superposition import ComplexSuperposition
from dense import ComplexDense
from mixture import ComplexMixture
from measurement import ComplexMeasurement

from utils import GetReal
from projection import Complex1DProjection
import math
import numpy as np
class QDNN(BasicModel):
    
    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='int32')
        
        self.phase_embedding=phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable)
        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init)
        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True)
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")        
        self.dropout = Dropout(self.opt.dropout_rate)
        
    def __init__(self,opt):
        super(QDNN, self).__init__(opt) 
        
    
    def build(self):
        self.weight = Activation('softmax')(self.weight_embedding(self.doc))
        self.phase_encoded = self.phase_embedding(self.doc)
        self.amplitude_encoded = self.amplitude_embedding(self.doc)
        
        if math.fabs(self.opt.dropout_rate -1) < 1e-6:
            self.phase_encoded = self.dropout(self.phase_encoded)
            self.amplitude_encoded = self.dropout(self.amplitude_encoded)
            
        [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([ self.phase_encoded, self.amplitude_encoded])
        if self.opt.network_type.lower() == 'complex_mixture':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])
    
        elif self.opt.network_type.lower() == 'complex_superposition':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, self.weight])
    
        else:
            print('Wrong input network type -- The default mixture network is constructed.')
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])
    
    
        probs = ComplexMeasurement(units = 12)([sentence_embedding_real, sentence_embedding_imag])
    
        output =  Dense(units = self.opt.nb_classes)(probs)
    
        model = Model(self.doc, output)
    

        return model