# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from keras.layers import  GlobalMaxPooling1D,    concatenate,Reshape, Permute

from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm

from layers.keras.complexnn import *

from keras.initializers import Constant
import numpy as np
import math

class Pooling(BasicModel):
    
    def initialize(self):
        pass
        
    def __init__(self,opt):
        super(Pooling, self).__init__(opt) 
        
    def get_representation(self,encoded):
        if math.fabs(self.opt.dropout_rate_probs-1) < 1e-6:
            encoded = self.dropout(encoded)
            
        representation = []
        for one_type in self.opt.pooling_type.split(','):
            if one_type == 'max':
                probs = GlobalMaxPooling1D()(encoded)
            elif one_type == 'average':
                probs = GlobalAveragePooling1D()(encoded)
            elif one_type == 'none':
                probs = Flatten()(encoded)
            elif one_type == 'max_col':
                probs = GlobalMaxPooling1D()(Permute((2,1))(encoded))
            elif one_type == 'average_col':
                probs = GlobalAveragePooling1D()(Permute((2,1))(encoded))
            else:
                print('Wrong input pooling type -- The default flatten layer is used.')
                probs = Flatten()(encoded)
            representation.append(probs)
        
        if len(representation)>1:
            representation = concatenate(representation)
        else:
            representation = representation[0]
        return representation