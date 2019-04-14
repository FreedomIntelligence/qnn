# -*- coding: utf-8 -*-
from models.BasicModel import BasicModel
from keras.layers import Embedding, GlobalMaxPooling1D,Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute,Lambda, Subtract
from keras.models import Model, Input, model_from_json, load_model, Sequential
from keras.constraints import unit_norm
from complexnn import *
import math
import numpy as np

from keras import regularizers
import keras.backend as K
from distutils.util import strtobool
from layers import Attention
from models import representation as representation_model_factory

class SiameseNetwork(BasicModel):

    def initialize(self):
        self.question = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.neg_answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
      
        distances= [getScore("AESD.AESD",mean="geometric",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="geometric",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="geometric",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("cosine.Cosinse",dropout_keep_prob =self.opt.dropout_rate_probs)
                    ]
                    
        self.distance = distances[self.opt.distance_type]

        
#        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))


        
        self.dense_last =  Dense(self.opt.nb_classes, activation="softmax")
                
    def __init__(self,opt):
        self.representation_model = representation_model_factory.setup(opt)
        super(SiameseNetwork, self).__init__(opt)
        

    def build(self):
        
        if self.opt.match_type == 'pointwise':
            reps = [self.representation_model.get_representation(doc) for doc in [self.question, self.answer]]
            if self.opt.onehot:
                output = self.dense_last(Attention()(reps))
            else:
                output = self.distance(reps)
#          
            model = Model([self.question,self.answer], output)
            
        elif self.opt.match_type == 'pairwise':

            q_rep = self.representation_model.get_representation(self.question)

            score1 = self.distance([q_rep, self.representation_model.get_representation(self.answer)])
            score2 = self.distance([q_rep, self.representation_model.get_representation(self.neg_answer)])
            basic_loss = MarginLoss(self.opt.margin)( [score1,score2])
            
            output=[score1,basic_loss,basic_loss]
            model = Model([self.question, self.answer, self.neg_answer], output)       
        else:
            raise ValueError('wrong input of matching type. Please input pairwise or pointwise.')
        return model
    
