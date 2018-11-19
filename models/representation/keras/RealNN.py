# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from modules.embedding.keras.RealEmbedding import RealEmbedding
from modules.encoding.keras.Pooling import Pooling
import math
class RealNN(BasicModel):
    
    def initialize(self):
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='int32')
        
        self.real_embedding= RealEmbedding(self.opt)
        self.pooling = Pooling(self.opt)
        
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")        
        
        
    def __init__(self,opt):
        super(RealNN, self).__init__(opt)        
        
    
    def build(self):      
        doc = Masking(mask_value = 0)(self.doc)
        embed = self.real_embedding.get_embedding(doc)
        encoded = self.pooling.get_representation(embed)
        output = self.dense(encoded)
        return Model(self.doc, output)


