
# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm

from layers.keras.complexnn import *

from keras.initializers import Constant
import numpy as np
import math

class Superposition(BasicModel):
    
    def initialize(self):
        pass
        
    def __init__(self,opt):
        super(Superposition, self).__init__(opt) 
    
    def get_representation(self,seq_embedding_real,seq_embedding_imag,weights,need_flatten =True):
        if weights is None:
            [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition(average_weights=True)([seq_embedding_real, seq_embedding_imag])
        else:
            [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, self.weight])
        if need_flatten:
            sentence_embedding_real = Flatten()(sentence_embedding_real)
            sentence_embedding_imag = Flatten()(sentence_embedding_imag)       
        return [sentence_embedding_real, sentence_embedding_imag]
