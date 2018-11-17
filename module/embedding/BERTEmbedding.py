# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from keras_bert import load_trained_model_from_checkpoint
from layers.keras.complexnn import *

from keras.initializers import Constant
import numpy as np
from module.embedding import ComplexWordEmbedding
class BERTEmbedding(ComplexWordEmbedding):
    
    def initialize(self):
       
#        self.phase_embedding=phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable)
#                
#        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init)
#
#        
#        checkpoint_path="D:/dataset/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" #chinese_L-12_H-768_A-12
#        config_path =   "D:/dataset/bert/uncased_L-12_H-768_A-12/bert_config.json" #chinese_L-12_H-768_A-12
#        dict_path =     "D:/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt" #chinese_L-12_H-768_A-12
        self.bertmodel = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)
        self.bertmodel.trainable = False
    def __init__(self,opt):
        super(ComplexWordEmbedding, self).__init__(opt) 
    
    def get_embedding(self,doc,mask,use_weight=True):

        amplitude_encoded = self.bertmodel([doc,mask]).output
        return self.process_complex_embedding(doc,amplitude_encoded,use_weight=use_weight)   
 
