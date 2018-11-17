# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers.keras.complexnn import *
import math
import numpy as np

from keras import regularizers

from module.embedding.ComplexWordEmbedding import ComplexWordEmbedding
from module.encoding.Mixture import Mixture

class QDNN(BasicModel):

    def initialize(self):
        self.doc = Input(shape=(self.opt.reader.max_sequence_length,), dtype='int32')
        self.complex_embedding_layer = ComplexWordEmbedding(self.opt)
        
        self.mixture_encoder = Mixture(self.opt)
        
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",

        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(QDNN, self).__init__(opt)


    def build(self):
        
        
        self.seq_embedding_real, self.seq_embedding_imag,self.word_weights = self.complex_embedding_layer.get_embedding(self.doc)
        sentence_embedding_real, sentence_embedding_imag = self.mixture_encoder.get_representation(self.seq_embedding_real,self.seq_embedding_imag,self.word_weights,need_flatten=False)

        if self.opt.network_type== "ablation" and self.opt.ablation == 1:
            sentence_embedding_real, sentence_embedding_imag = self.mixture_encoder.get_representation(self.seq_embedding_real,self.seq_embedding_imag,self.word_weights,need_flatten=False)

            predictions = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", init_criterion = self.opt.init_mode)([sentence_embedding_real, sentence_embedding_imag])
            output = GetReal()(predictions)

        else:
#            cancel the above operation, if needed.
#            sentence_embedding_real = Flatten()(sentence_embedding_real)
#            sentence_embedding_imag = Flatten()(sentence_embedding_imag)
            sentence_embedding_real, sentence_embedding_imag = self.mixture_encoder.get_representation(self.seq_embedding_real,self.seq_embedding_imag,self.word_weights,need_flatten=False)

            probs =  self.projection([sentence_embedding_real, sentence_embedding_imag])
            if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
                probs = self.dropout_probs(probs)
            output = self.dense(probs)
        model = Model(self.doc, output)
        return model


