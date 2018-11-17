# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute

from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers.keras.complexnn import *
import math
import numpy as np
from models.embedding.ComplexWordEmbedding import ComplexWordEmbedding


from keras import regularizers
import keras.backend as K

from models.BasicModel import BasicModel
class Ngram(BasicModel):

    def initialize(self):
        self.n_value = int(self.opt.ngram_value)
#        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
#        #############################################
#        #This parameter should be passed from params
##        self.ngram = NGram(n_value = self.opt.ngram_value)
#        self.ngram = [NGram(n_value = int(n_value)) for n_value in self.opt.ngram_value.split(',')]
#        #############################################
#        self.phase_embedding= phase_embedding_layer(None, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)
#
#        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), None, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
#        self.complex_embedding_layer = ComplexWordEmbedding(self.opt)
#
#        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True, input_length = None)
#        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
#        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
#        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
#        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(Ngram, self).__init__(opt)


#    def build(self):
#        self.probs = self.get_representation(self.doc)
#        output = self.dense(self.probs)
#        model = Model(self.doc, output)
#        return model
    
    def get_representation(self,seq_embedding_real,seq_embedding_imag):
        
        seq_embedding_real = NGram(n_value = self.n_value)(seq_embedding_real)
        seq_embedding_imag = NGram(n_value = self.n_value)(seq_embedding_imag)
        
        return seq_embedding_real, seq_embedding_imag
    



