# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
import sys
from .QDNN import QDNN
from layers.keras.complexnn import *

import math
import numpy as np

from keras import regularizers
from keras.initializers import Constant

from keras.models import Sequential



projector_to_dense = 1
projector_without_training = 2
amplitude_embedding_without_training =3
word_weight_without_training =4
word_weigth_with_idf = 5

class QDNNAblation(QDNN):
    def initialize(self):
        super(QDNNAblation, self).initialize()
        self.ablation()
    def __init__(self,opt):
        super(QDNNAblation, self).__init__(opt)

        
    def ablation(self):
        if self.opt.ablation== projector_to_dense:
            print("projector_to_dense")
            self.projection = ComplexDense(units = self.opt.nb_classes, activation= "sigmoid", bias_initializer=Constant(value=-1), init_criterion = self.opt.init_mode)
        elif self.opt.ablation == projector_without_training:
            print("projector_without_training")
            
            self.projection = ComplexMeasurement(units = self.opt.measurement_size,trainable = False)
        elif self.opt.ablation == amplitude_embedding_without_training:
            print("amplitude_embedding_without_training")
            self.complex_embedding_layer.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = False, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
        elif self.opt.ablation == word_weight_without_training:
            print("word_weight_without_training")
            self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = False)
        elif self.opt.ablation == word_weigth_with_idf:
            weights= np.array([[num] for num in self.opt.idfs])
            print(weights.shape)
#            print(self.opt.lookup_table.shape[0], 1)
            self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = False,weights=[weights])
        else:
            pass
            