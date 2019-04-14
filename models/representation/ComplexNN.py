# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
import sys
from complexnn import *

from keras.initializers import Constant
import math
import numpy as np
class ComplexNN(BasicModel):
    
    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='int32')
        
        self.phase_embedding=phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable)
        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init)
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")        
        self.dropout = Dropout(self.opt.dropout_rate_probs)
        
    def __init__(self,opt):
        super(ComplexNN, self).__init__(opt) 
        
    
    def build(self):
        [sentence_embedding_real,sentence_embedding_imag] = self.get_representation(self.doc)
    # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
        predictions = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", bias_initializer=Constant(value=-1), init_criterion = self.opt.init_mode)([sentence_embedding_real, sentence_embedding_imag])
        output = GetReal()(predictions) 
        model = Model(self.doc, output)
        return model
    
    def get_representation(self,doc):
        self.phase_encoded = self.phase_embedding(doc)
        self.amplitude_encoded = self.amplitude_embedding(doc)
        
        if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
            self.phase_encoded = self.dropout(self.phase_encoded)
            self.amplitude_encoded = self.dropout(self.amplitude_encoded)
            
        [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([ self.phase_encoded, self.amplitude_encoded])
        if self.opt.network_type.lower() == 'complex_mixture':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture(average_weights=True)([seq_embedding_real, seq_embedding_imag])
    
        elif self.opt.network_type.lower() == 'complex_superposition':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition(average_weights=True)([seq_embedding_real, seq_embedding_imag])
    
        else:
            print('Wrong input network type -- The default mixture network is constructed.')
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture(average_weights=True)([seq_embedding_real, seq_embedding_imag])
    
    
        sentence_embedding_real = Flatten()(sentence_embedding_real)
        sentence_embedding_imag = Flatten()(sentence_embedding_imag)       
        return [sentence_embedding_real, sentence_embedding_imag]