# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute,Lambda
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from keras_bert import load_trained_model_from_checkpoint
import math
import os
class BERTFastext(BasicModel):
    
    def initialize(self):
        self.bert_dir = self.opt.bert_dir  
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")        
        self.dropout = Dropout(self.opt.dropout_rate_probs)       
     
        self.bertmodel = self.get_BERT_model()
        self.bertmodel.trainable = False        
    
    def __init__(self,opt):
        super(BERTFastext, self).__init__(opt)
            
        
    
    def build(self):        
        output = self.dense(self.get_representation(self.bertmodel.output))
        return Model(self.bertmodel.input, output)
    
    def get_representation(self,encoded):       

        representation = []
        for one_type in self.opt.pooling_type.split(','):
            if self.opt.pooling_type == 'max':
                probs = Lambda(lambada_max)(encoded)
            elif self.opt.pooling_type == 'average':
                probs = Lambda(lambada_mean)(encoded)
            elif self.opt.pooling_type == 'none':
                probs = Flatten()(encoded)
            elif self.opt.pooling_type == 'max_col':
                probs = GlobalMaxPooling1D()(Permute((2,1))(encoded))
            elif self.opt.pooling_type == 'average_col':
                probs = GlobalAveragePooling1D()(Permute((2,1))(encoded))
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = Flatten()(encoded)
            representation.append(probs)
        
        if len(representation)>1:
            representation = concatenate(representation)
        else:
            representation = representation[0]
#        representation =GlobalAveragePooling1D()(self.encoded)
        return(representation)
        
    def get_BERT_model(self):
        checkpoint_path = os.path.join(self.bert_dir,'bert_model.ckpt')
        config_path = os.path.join(self.bert_dir,'bert_config.json')
        return load_trained_model_from_checkpoint(config_path, checkpoint_path)

def lambada_mean(x):
    return K.mean(x, axis=1)
def lambada_max(x):
    return K.max(x, axis=1)


        
import keras.backend as K        
class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask != None:
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return super().call(x)

