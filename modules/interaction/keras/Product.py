# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from keras.layers import Multiply

from layers.keras.complexnn import *

from keras.initializers import Constant
import numpy as np
import math

class Product(BasicModel):
    
    def initialize(self):
        pass
        
    def __init__(self,opt):
        super(Product, self).__init__(opt) 
    
    def get_representation(self, rep_left, rep_right):
        if type(rep_left) is list: 
            # complex product
            [output_real, output_imag] = ComplexProduct()([rep_left, rep_right])
            output = [output_real, output_imag]
        else:
            # real product
            output = Multiply()([rep_left, rep_right])

        return output
