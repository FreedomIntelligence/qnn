# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm

from layers.keras.complexnn import *
from keras_bert import load_trained_model_from_checkpoint
from keras.initializers import Constant
import numpy as np
import math
from distutils.util import strtobool
import os

class ComplexWordEmbedding(BasicModel):
    
    def initialize(self):    
        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), None, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
        self.phase_embedding= phase_embedding_layer(None, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)
        
        self.l2_normalization = L2Normalization(axis = -1)
        self.l2_norm = L2Norm(axis = -1, keep_dims = False)
#        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True, input_length = None)
#        self.weight = Activation('softmax')(self.weight_embedding(doc))
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        if self.opt.bert_enabled:
            checkpoint_path = os.path.join(self.opt.bert_dir,'bert_model.ckpt')
            config_path = os.path.join(self.opt.bert_dir,'bert_config.json')
            self.bertmodel = load_trained_model_from_checkpoint(config_path, checkpoint_path)
            self.remove_mask = RemoveMask()
        
    def __init__(self,opt):
        super(ComplexWordEmbedding, self).__init__(opt) 
    
        
    def process_complex_embedding(self,phase_encoded,amplitude_encoded,use_weight=False):
        if use_weight:
#            self.weight = Activation('softmax')(self.l2_norm(amplitude_encoded))
            self.weight = self.l2_norm(amplitude_encoded)
#            print(self.weight.shape)
#            print(self.weight.shape)
#            self.weight = reshape((-1,self.opt.max_sequence_length,self.opt.ngram_value,1))(self.weight)
            self.amplitude_encoded = self.l2_normalization(amplitude_encoded)  
#            print(self.amplitude_encoded.shape)
        else:
            self.weight = None
            
        if math.fabs(self.opt.dropout_rate_probs-1) < 1e-6:
            self.phase_encoded = self.dropout_embedding(phase_encoded)
            self.amplitude_encoded = self.dropout_embedding(amplitude_encoded)
            
        [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_encoded, amplitude_encoded])
        return seq_embedding_real,seq_embedding_imag,self.weight
        
    def get_embedding(self,doc,use_weight=False):
        if self.opt.bert_enabled:
            amplitude_encoded = self.bertmodel([doc[0], doc[1]])
            amplitude_encoded = self.remove_mask(amplitude_encoded)
            print(amplitude_encoded.shape)
            
            self.phase_embedding = phase_embedding_layer(None, int(amplitude_encoded.shape[1]), int(amplitude_encoded.shape[2]), trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)
            phase_encoded = self.phase_embedding(doc[0])
#            print(phase_encoded.shape)
        else:
            phase_encoded = self.phase_embedding(doc)
            amplitude_encoded = self.amplitude_embedding(doc)
        
            
        return self.process_complex_embedding(phase_encoded,amplitude_encoded,use_weight=use_weight)

