# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from keras_bert import load_trained_model_from_checkpoint
from layers.keras.complexnn import *

from keras.initializers import Constant
import numpy as np
from modules.embedding.keras.ComplexWordEmbedding import ComplexWordEmbedding
import os
class BERTEmbedding(ComplexWordEmbedding):
    
    def initialize(self):
        super(BERTEmbedding, self).initialize() 
#        self.phase_embedding=phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable)
#                
#        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init)
#
#        
#        checkpoint_path="D:/dataset/bert/uncased_L-12_H-768_A-12/bert_model.ckpt" #chinese_L-12_H-768_A-12
#        config_path =   "D:/dataset/bert/uncased_L-12_H-768_A-12/bert_config.json" #chinese_L-12_H-768_A-12
#        dict_path =     "D:/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt" #chinese_L-12_H-768_A-12
        checkpoint_path = os.path.join(self.opt.bert_dir,'bert_model.ckpt')
        config_path = os.path.join(self.opt.bert_dir,'bert_config.json')
        self.bertmodel = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        self.remove_mask = RemoveMask()
#        self.bertmodel.trainable = False
    def __init__(self,opt):
        super(BERTEmbedding, self).__init__(opt) 
    
    def get_embedding(self,doc,mask,use_complex=False):

        amplitude_encoded = self.bertmodel([doc, mask])
        amplitude_encoded = self.remove_mask(amplitude_encoded)
        self.phase_embedding = phase_embedding_layer(None, int(amplitude_encoded.shape[1]), int(amplitude_encoded.shape[2]), trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)
        if use_complex:
            return  super(BERTEmbedding, self).process_complex_embedding(doc,amplitude_encoded,use_weight=True)   
        else:
            return amplitude_encoded
 
