# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers.keras.complexnn import *
import math
import numpy as np

from keras import regularizers

from models.embedding.ComplexWordEmbedding import ComplexWordEmbedding


class QDNN(BasicModel):

    def initialize(self):
        self.doc = Input(shape=(self.opt.reader.max_sequence_length,), dtype='int32')
        self.complex_embedding_layer = ComplexWordEmbedding(self.opt)
        self.l2_normalization = L2Normalization(axis = 2)
        self.l2_norm = L2Norm(axis = 2, keep_dims = False)
        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True, input_length = None)
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(QDNN, self).__init__(opt)


    def build(self):
        probs = self.get_representation(self.doc)
        if self.opt.network_type== "ablation" and self.opt.ablation == 1:
            predictions = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", init_criterion = self.opt.init_mode)(probs)
            output = GetReal()(predictions)
        else:
            output = self.dense(probs)
        model = Model(self.doc, output)
        return model
    
    def get_representation(self,doc):
        
        self.amplitude_encoded,self.phase_encoded = self.complex_embedding_layer.get_embedding(doc)
#        self.phase_encoded = self.phase_embedding(doc)
#        self.amplitude_encoded = self.amplitude_embedding(doc)
        self.weight = Activation('softmax')(self.l2_norm(self.amplitude_encoded))
        self.amplitude_encoded = self.l2_normalization(self.amplitude_encoded)
#        self.weight = Activation('softmax')(self.weight_embedding(doc))
#        self.phase_encoded = self.phase_embedding(doc)
#        self.amplitude_encoded = self.amplitude_embedding(doc)

        if math.fabs(self.opt.dropout_rate_embedding -1) < 1e-6:
            self.phase_encoded = self.dropout_embedding(self.phase_encoded)
            self.amplitude_encoded = self.dropout_embedding(self.amplitude_encoded)

        [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([self.phase_encoded, self.amplitude_encoded])
        if self.opt.network_type.lower() == 'complex_mixture':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])

        elif self.opt.network_type.lower() == 'complex_superposition':
            [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, self.weight])

        else:
            print('Wrong input network type -- The default mixture network is constructed.')
            [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, self.weight])
        if self.opt.network_type== "ablation" and self.opt.ablation == 1:
            sentence_embedding_real = Flatten()(sentence_embedding_real)
            sentence_embedding_imag = Flatten()(sentence_embedding_imag)
            probs = [sentence_embedding_real, sentence_embedding_imag]
        # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
        else:
            probs =  self.projection([sentence_embedding_real, sentence_embedding_imag])
#            print(probs.shape)
            if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
                probs = self.dropout_probs(probs)
        return(probs)


