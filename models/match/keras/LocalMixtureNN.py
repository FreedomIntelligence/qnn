# -*- coding: utf-8 -*-
from models.match.keras.BasicModel import BasicModel
from keras.layers import Embedding, GlobalMaxPooling1D,Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute,Lambda, Subtract
from keras.models import Model, Input, model_from_json, load_model, Sequential
from keras.constraints import unit_norm
from complexnn import *
import math
import numpy as np

from keras import regularizers
import keras.backend as K
from models.representation.LocalMixtureNN import  LocalMixtureNN as rep_model

class LocalMixtureNN(BasicModel):

    def initialize(self):
        self.question = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.neg_answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.ngram =  NGram(n_value = self.opt.ngram_value) 
        self.phase_embedding= phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)
        self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
        self.l2_normalization = L2Normalization(axis = 3)
        self.l2_norm = L2Norm(axis = 3, keep_dims = False)
        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True)
        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        self.dropout_probs = Dropout(1) #self.opt.dropout_rate_probs
        self.projection = ComplexMeasurement(units = self.opt.measurement_size)
#        self.distance = Lambda(l2_distance)
#        self.distance = Lambda(cosine_similarity)
#        self.triplet_loss = Lambda(triplet_hinge_loss)
        distances= [getScore("AESD.AESD",mean="geometric",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="geometric",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="geometric",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("cosine.Cosinse",dropout_keep_prob =self.opt.dropout_rate_probs),
                    ]
                    
        self.distance= distances[self.opt.distance_type]
        
#        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))
                
    def __init__(self,opt):
        super(LocalMixtureNN, self).__init__(opt)

    def build(self):
        rep_m = rep_model(self.opt)
        if self.opt.match_type == 'pointwise':
            rep = []
            for doc in [self.question, self.answer]:
                rep.append(rep_m.get_representation(doc))
            output = self.distance(rep)
#            output =  Cosinse(dropout_keep_prob=self.opt.dropout_rate_probs)(rep) 
#            output = AESD()(rep)
            model = Model([self.question, self.answer], output)
        elif self.opt.match_type == 'pairwise':
#            rep = []
#            for doc in [self.question, self.answer, self.neg_answer]:
#                rep.append(rep_m.get_representation(doc))
            q_rep = self.dropout_probs(rep_m.get_representation(self.question))
            
            score1 = self.distance([q_rep, rep_m.get_representation(self.answer)])
            score2 = self.distance([q_rep, rep_m.get_representation(self.neg_answer)])
            basic_loss = MarginLoss(self.opt.margin)( [score1,score2])
            
            output=[score1,basic_loss,basic_loss]
            model = Model([self.question, self.answer, self.neg_answer], output)           
        else:
            raise ValueError('wrong input of matching type. Please input pairwise or pointwise.')
        return model
        


if __name__ == "__main__":

    from params import Params
    import numpy as np
    from dataset import qa
    import units
    opt = Params()
    config_file = 'config/qalocal_point.ini'    # define dataset in the config
    opt.parse_config(config_file)
    reader = qa.setup(opt)
    self = BasicModel(opt)
#    model.compile(loss = opt.loss,
#            optimizer = units.getOptimizer(name=opt.optimizer,lr=opt.lr),
#            metrics=['accuracy'])
#    print(model.predict(x = [train_x,train_x]))
#    

