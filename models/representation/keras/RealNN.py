# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute,Lambda
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from modules.embedding.keras.RealEmbedding import RealEmbedding
from modules.encoding.keras.Pooling import Pooling
import math
import keras.backend as K
class RealNN(BasicModel):
    
    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='int32')
        
        self.embedding_module= RealEmbedding(self.opt)
        self.pooling = Pooling(self.opt)
        
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")        
        
        
    def __init__(self,opt):
        super(RealNN, self).__init__(opt)        
        
    
#    def build(self):      
#        encoded = self.get_representation(self.doc)
#        output = self.dense(encoded)
#        return Model(self.doc, output)
    
    def get_representation(self,doc):
        embedded = self.embedding_module.get_embedding(doc,use_weight=False)
        representation = []
        for one_type in self.opt.pooling_type.split(','):
            if self.opt.pooling_type == 'max':
                probs = Lambda(lambda_max)(embedded)
            elif self.opt.pooling_type == 'average':
                probs = Lambda(lambda_mean)(embedded)
            elif self.opt.pooling_type == 'none':
                probs = Flatten()(embedded)
            elif self.opt.pooling_type == 'max_col':
                probs = GlobalMaxPooling1D()(Permute((2,1))(embedded))
            elif self.opt.pooling_type == 'average_col':
                probs = GlobalAveragePooling1D()(Permute((2,1))(embedded))
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = Flatten()(embedded)
            representation.append(probs)
        
        if len(representation)>1:
            representation = concatenate(representation)
        else:
            representation = representation[0]
#        representation =GlobalAveragePooling1D()(self.encoded)
        return(representation)
        
#        embed = self.real_embedding.get_embedding(masked_doc)
#        encoded = self.pooling.get_representation(embed)
#        return 
         

def lambda_mean(x):
    return K.mean(x, axis=1)
def lambda_max(x):
    return K.max(x, axis=1)

