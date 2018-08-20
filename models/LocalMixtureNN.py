# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation,concatenate
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
from l2_normalization import L2Normalization
from l2_norm import L2Norm
from index import Index
from ngram import NGram
from utils import GetReal
from projection import Complex1DProjection
import math
import numpy as np

from keras import regularizers



class LocalMixtureNN(BasicModel):

    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        #############################################
        #This parameter should be passed from params
        self.ngram = NGram(n_value = 3)
        #############################################
        self.phase_embedding=phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)

        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
        self.l2_normalization = L2Normalization(axis = 2)
        self.l2_norm = L2Norm(axis = 2)
        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True)
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(LocalMixtureNN, self).__init__(opt)


    def build(self):

        self.doc_ngram = self.ngram(self.doc)
        self.prob_mat = []
        for i in range(self.opt.max_sequence_length):

            # print(i)
            self.input_i = Index(i)(self.doc_ngram)
            self.amplitude_encoded = self.amplitude_embedding(self.input_i)
            self.weight = Activation('softmax')(self.l2_norm(self.amplitude_encoded))
            self.amplitude_encoded = self.l2_normalization(self.amplitude_encoded)
            # print(self.weight.shape)
            self.phase_encoded = self.phase_embedding(self.input_i)

            # print(self.amplitude_encoded.shape)
            if math.fabs(self.opt.dropout_rate_embedding -1) < 1e-6:
                self.phase_encoded = self.dropout_embedding(self.phase_encoded)
                self.amplitude_encoded = self.dropout_embedding(self.amplitude_encoded)

            [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([ self.phase_encoded, self.amplitude_encoded])
            if self.opt.network_type.lower() == 'complex_mixture':
                [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])

            elif self.opt.network_type.lower() == 'complex_superposition':
                [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, self.weight])

            else:
                print('Wrong input network type -- The default mixture network is constructed.')
                [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])

            probs =  self.projection([sentence_embedding_real, sentence_embedding_imag])

            if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
                probs = self.dropout_probs(probs)
                # output =  self.dense(probs)
            self.prob_mat.append(probs)

        self.prob_flattened = concatenate(self.prob_mat)
        output = self.dense(self.prob_flattened)
        model = Model(self.doc, output)
        return model


