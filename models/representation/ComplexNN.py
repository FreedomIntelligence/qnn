# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
import sys
from layers.keras.complexnn import *
from models.embedding.ComplexWordEmbedding import ComplexWordEmbedding

from keras.initializers import Constant
import math
import numpy as np
class ComplexNN(BasicModel):
    
    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='int32')
        self.complex_embedding_layer = ComplexWordEmbedding(self.opt)
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
        self.amplitude_encoded,self.phase_encoded = self.complex_embedding_layer.get_embedding(doc)

        
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