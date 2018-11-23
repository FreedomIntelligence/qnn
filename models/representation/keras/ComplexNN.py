# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
import sys
from layers.keras.complexnn import *
from modules.embedding.keras.ComplexWordEmbedding import ComplexWordEmbedding
from modules.encoding.keras.Mixture import Mixture
from keras.initializers import Constant
import math
import numpy as np

class ComplexNN(BasicModel):
    
    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='int32')
        self.complex_embedding_layer = ComplexWordEmbedding(self.opt)
        self.mixture_encoder = Mixture(self.opt)
        self.dense = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", bias_initializer=Constant(value=-1), init_criterion = self.opt.init_mode)     

        
    def __init__(self,opt):
        super(ComplexNN, self).__init__(opt) 
        
    
#    def build(self):
#
#
#        sentence_embedding_real, sentence_embedding_imag = self.get_representation(self.doc)
#    # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
#        predictions =self.dense([sentence_embedding_real, sentence_embedding_imag])
#        output = GetReal()(predictions) 
#        model = Model(self.doc, output)
#        return model
    
    def get_representation(self,doc):
        self.seq_embedding_real, self.seq_embedding_imag,self.word_weights = self.complex_embedding_layer.get_embedding(doc)
        if not self.word_weights is None:
            self.word_weights = Activation('softmax')(self.word_weights)
        sentence_embedding_real, sentence_embedding_imag = self.mixture_encoder.get_representation(self.seq_embedding_real,self.seq_embedding_imag,self.word_weights)
        predictions =self.dense([sentence_embedding_real, sentence_embedding_imag])
        return predictions