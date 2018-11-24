# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from keras_bert import load_trained_model_from_checkpoint
from layers.keras.complexnn import *

from keras.initializers import Constant
import numpy as np
import math
import os
import keras.backend as K

class RealEmbedding(BasicModel):
    
    def initialize(self):
#        
        if(self.opt.random_init):
            self.embedding = Embedding(trainable=self.opt.embedding_trainable, input_dim=self.opt.lookup_table.shape[0],output_dim=self.opt.lookup_table.shape[1], 
                                    weights=[self.opt.lookup_table],embeddings_constraint = unit_norm(axis = 1))
        else:
            self.embedding = Embedding(trainable=self.opt.embedding_trainable, input_dim=self.opt.lookup_table.shape[0],output_dim=self.opt.lookup_table.shape[1],embeddings_constraint = unit_norm(axis = 1))
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        if self.opt.bert_enabled:
            checkpoint_path = os.path.join(self.opt.bert_dir,'bert_model.ckpt')
            config_path = os.path.join(self.opt.bert_dir,'bert_config.json')
            self.bertmodel = load_trained_model_from_checkpoint(config_path, checkpoint_path, training = False)
            self.bertmodel.trainable = False
            self.remove_mask = RemoveMask()

    def __init__(self,opt):
        super(RealEmbedding, self).__init__(opt) 
            
    def get_embedding(self,doc,use_weight=False):
        if self.opt.bert_enabled:
            encoded= self.bertmodel([doc[0],doc[1]])
            encoded = self.remove_mask(encoded)
        else:
            encoded = self.embedding(doc)
        if math.fabs(self.opt.dropout_rate_probs -1) < 1e-6:
            encoded = self.dropout_embedding(encoded)
        
        return encoded

