# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers.keras.complexnn import *
from distutils.util import strtobool
import math
import numpy as np

from keras import regularizers

from modules.embedding.keras.ComplexWordEmbedding import ComplexWordEmbedding
from modules.encoding.keras.Mixture import Mixture
from modules.embedding.keras.BERTEmbedding import BERTEmbedding


class QDNN(BasicModel):

    def initialize(self):
        self.doc = Input(shape=(self.opt.reader.max_sequence_length,), dtype='int32')
       
        self.embedding_module = ComplexWordEmbedding(self.opt)
            
        self.encoding_module = Mixture(self.opt)
        
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",

        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(QDNN, self).__init__(opt)

    
    def get_representation(self,doc):
        self.seq_embedding_real, self.seq_embedding_imag, self.word_weights = self.embedding_module.get_embedding(doc,use_weight=True)
        self.word_weights = Activation('softmax')(self.word_weights)
#        sentence_embedding_real, sentence_embedding_imag = self.encoding_module.get_representation(self.seq_embedding_real,self.seq_embedding_imag,self.word_weights,need_flatten=False)

        if self.opt.network_type== "ablation" and self.opt.ablation == 1:
            sentence_embedding_real, sentence_embedding_imag = self.encoding_module.get_representation(self.seq_embedding_real,self.seq_embedding_imag,self.word_weights,need_flatten=False)

            predictions = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", init_criterion = self.opt.init_mode)([sentence_embedding_real, sentence_embedding_imag])
            output = GetReal()(predictions)

        else:
#            cancel the above operation, if needed.
#            sentence_embedding_real = Flatten()(sentence_embedding_real)
#            sentence_embedding_imag = Flatten()(sentence_embedding_imag)
            sentence_embedding_real, sentence_embedding_imag = self.encoding_module.get_representation(self.seq_embedding_real,self.seq_embedding_imag,self.word_weights,need_flatten=False)

            probs =  self.projection([sentence_embedding_real, sentence_embedding_imag])
            if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
                probs = self.dropout_probs(probs)
            output = probs
                
        return output

