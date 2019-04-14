# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm

import math
class RealNN(BasicModel):
    
    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='int32')
        
        if(self.opt.random_init):
            self.embedding = Embedding(trainable=self.opt.embedding_trainable, input_dim=self.opt.lookup_table.shape[0],output_dim=self.opt.lookup_table.shape[1], 
                                    weights=[self.opt.lookup_table],embeddings_constraint = unit_norm(axis = 1))
        else:
            self.embedding = Embedding(trainable=self.opt.embedding_trainable, input_dim=self.opt.lookup_table.shape[0],output_dim=self.opt.lookup_table.shape[1],embeddings_constraint = unit_norm(axis = 1))
        
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")        
        self.dropout = Dropout(self.opt.dropout_rate_probs)
        
    def __init__(self,opt):
        super(RealNN, self).__init__(opt)        
        
    
    def build(self):        
        representation = self.get_representation(self.doc)
        output = self.dense(representation)
        return Model(self.doc, output)
    
    def get_representation(self,doc):
        doc = Masking(mask_value = 0)(doc)
        self.encoded = self.embedding(doc)
        if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
            self.encoded = self.dropout(self.encoded)
        representation = []
        for one_type in self.opt.pooling_type.split(','):
            if self.opt.pooling_type == 'max':
                probs = GlobalMaxPooling1D()(self.encoded)
            elif self.opt.pooling_type == 'average':
                probs = GlobalAveragePooling1D()(self.encoded)
            elif self.opt.pooling_type == 'none':
                probs = Flatten()(self.encoded)
            elif self.opt.pooling_type == 'max_col':
                probs = GlobalMaxPooling1D()(Permute((2,1))(self.encoded))
            elif self.opt.pooling_type == 'average_col':
                probs = GlobalAveragePooling1D()(Permute((2,1))(self.encoded))
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = Flatten()(self.encoded)
            representation.append(probs)
        
        if len(representation)>1:
            representation = concatenate(representation)
        else:
            representation = representation[0]
#        representation =GlobalAveragePooling1D()(self.encoded)
        return(representation)


