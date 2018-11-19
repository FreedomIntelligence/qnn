# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute,Lambda
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from keras_bert import load_trained_model_from_checkpoint
import math
import os

from modules.embedding.keras.BERTEmbedding import  BERTEmbedding
class BERTFastext(BasicModel):
    
    def initialize(self):
        self.bert_dir = self.opt.bert_dir  
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")        
        
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.mask = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        
        self.bert_embedding = BERTEmbedding(self.opt)
#        self.pooling = Pooling(self.opt)
    
    def __init__(self,opt):
        super(BERTFastext, self).__init__(opt)
            
        
    
    def build(self):  

        encoded = self.get_representation(self.doc,self.mask)  #self.pooling does not work here
        output = self.dense(encoded)
        return Model([self.doc,self.mask], output)
        
#        return Model(self.bertmodel.input, output)
    
    def get_representation(self,doc,mask=None):       
        embed = self.bert_embedding.get_embedding(doc,mask,use_complex=False)
        representation = []
        for one_type in self.opt.pooling_type.split(','):
            if self.opt.pooling_type == 'max':
                probs = Lambda(lambda_max)(embed)
            elif self.opt.pooling_type == 'average':
                probs = Lambda(lambda_mean)(embed)
            elif self.opt.pooling_type == 'none':
                probs = Flatten()(embed)
            elif self.opt.pooling_type == 'max_col':
                probs = GlobalMaxPooling1D()(Permute((2,1))(embed))
            elif self.opt.pooling_type == 'average_col':
                probs = GlobalAveragePooling1D()(Permute((2,1))(embed))
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = Flatten()(embed)
            representation.append(probs)
        
        if len(representation)>1:
            representation = concatenate(representation)
        else:
            representation = representation[0]
#        representation =GlobalAveragePooling1D()(self.encoded)
        return(representation)
        


import keras.backend as K 
def lambda_mean(x):
    return K.mean(x, axis=1)
def lambda_max(x):
    return K.max(x, axis=1)


#        
#import keras.backend as K        
#class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
#    def call(self, x, mask=None):
#        if mask != None:
#            return K.sum(x, axis=1) / K.sum(mask, axis=1)
#        else:
#            return super().call(x)

