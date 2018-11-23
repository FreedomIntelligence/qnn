# -*- coding: utf-8 -*-
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation
from models.BasicModel import BasicModel
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers.keras.complexnn import *
import math
import numpy as np

from keras import regularizers

from models.representation.keras.ComplexNN import ComplexNN as rep_model

class ComplexNN(BasicModel):

    def initialize(self):
        self.doc = Input(shape=(self.opt.reader.max_sequence_length,), dtype='int32')
        if self.opt.bert_enabled:
            self.mask = Input(shape=(self.opt.reader.max_sequence_length,), dtype='int32')
            self.doc = [self.doc,self.mask]
#        self.mask = Input(shape=(self.opt.reader.max_sequence_length,), dtype='int32')
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",

    def __init__(self,opt):
        super(ComplexNN, self).__init__(opt)


    def build(self):
        rep_m = rep_model(self.opt)
        representation = rep_m.get_representation(self.doc)
        output = self.dense(representation)
        model = Model(self.doc, output)
        return model