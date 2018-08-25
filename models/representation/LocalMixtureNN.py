# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute
from .BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from complexnn import *
import math
import numpy as np

from keras import regularizers
import keras.backend as K


class LocalMixtureNN(BasicModel):

    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        #############################################
        #This parameter should be passed from params
        self.ngram =  NGram(n_value = self.opt.ngram_value) 
        #############################################
        self.phase_embedding= phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)

        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
        self.l2_normalization = L2Normalization(axis = 3)
        self.l2_norm = L2Norm(axis = 3, keep_dims = False)
        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True)
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
        self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    def __init__(self,opt):
        super(LocalMixtureNN, self).__init__(opt)


    def build(self):
        self.probs = self.get_representation(self.doc)
        output = self.dense(self.probs)
        model = Model(self.doc, output)
        return model
    
    def get_representation(self,doc):
        self.doc_ngram = self.ngram(doc)

        self.inputs = [Index(i)(self.doc_ngram) for i in range(self.opt.max_sequence_length)]
        self.inputs_count = len(self.inputs)
        self.inputs = concatenate(self.inputs)

        self.phase_encoded = self.phase_embedding(self.inputs)
        self.amplitude_encoded = self.amplitude_embedding(self.inputs)
        
        self.phase_encoded = reshape((-1,self.opt.max_sequence_length,self.opt.ngram_value,self.opt.lookup_table.shape[1]))(self.phase_encoded)
        self.amplitude_encoded = reshape((-1,self.opt.max_sequence_length,self.opt.ngram_value,self.opt.lookup_table.shape[1]))(self.amplitude_encoded)
#        self.weight = Activation('softmax')(self.weight_embedding(self.inputs))
        
        self.weight = Activation('softmax')(self.l2_norm(self.amplitude_encoded))
        self.weight = reshape((-1,self.opt.max_sequence_length,self.opt.ngram_value,1))(self.weight)
        self.amplitude_encoded = self.l2_normalization(self.amplitude_encoded)
        
        
#        self.weight = reshape( (-1,self.opt.max_sequence_length,self.opt.ngram_value))(self.weight)
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

        probs =  self.projection([sentence_embedding_real, sentence_embedding_imag])
#        probs = Permute((2,1))(probs)
#        self.probs = reshape((-1,self.opt.max_sequence_length*self.opt.measurement_size))(probs)
        if self.opt.pooling_type == 'max':
            probs = GlobalMaxPooling1D()(probs)
        elif self.opt.pooling_type == 'average':
            probs = GlobalAveragePooling1D()(probs)
        elif self.opt.pooling_type == 'none':
            probs = Flatten()(probs)
        else:
            print('Wrong input pooling type -- The default flatten layer is used.')
            probs = Flatten()(probs)
        if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
            probs = self.dropout_probs(probs)

        
        return(probs)
    


if __name__ == "__main__":
    from models.BasicModel import BasicModel
    from params import Params
    import dataset
    opt = Params()
    config_file = 'config/local.ini'    # define dataset in the config
    opt.parse_config(config_file)
    reader = dataset.setup(opt)
    opt = dataset.process_embedding(reader,opt)
    (train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()
    self = BasicModel(opt)

