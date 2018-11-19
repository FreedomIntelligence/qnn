# -*- coding: utf-8 -*-
from models.BasicModel import BasicModel
from keras.layers import Embedding, GlobalMaxPooling1D,Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute,Lambda, Subtract
from keras.models import Model, Input, model_from_json, load_model, Sequential
from keras.constraints import unit_norm
from layers.keras.complexnn import *
import math
import numpy as np

from keras import regularizers
import keras.backend as K
from models.representation.keras.RealNN import RealNN as rep_model

class RealNN(BasicModel):

    def initialize(self):
        self.question = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.neg_answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
         
        if(self.opt.random_init):
            self.embedding = Embedding(trainable=self.opt.embedding_trainable, input_dim=self.opt.lookup_table.shape[0],output_dim=self.opt.lookup_table.shape[1], 
                                    weights=[self.opt.lookup_table],embeddings_constraint = unit_norm(axis = 1))
        else:
            self.embedding = Embedding(trainable=self.opt.embedding_trainable, input_dim=self.opt.lookup_table.shape[0],output_dim=self.opt.lookup_table.shape[1],embeddings_constraint = unit_norm(axis = 1))
        
        self.dense = Dense(self.opt.nb_classes, activation="sigmoid")       
        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
#        self.distance = Lambda(l2_distance)
        distances= [getScore("AESD.AESD",mean="geometric",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="geometric",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="geometric",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("AESD.AESD",mean="arithmetic",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    getScore("cosine.Cosinse",dropout_keep_prob =self.opt.dropout_rate_probs)
                    ]
                    
        self.distance= distances[self.opt.distance_type]
        if self.opt.onehot:
            self.distance = getScore("multiple_loss.Multiple_loss",dropout_keep_prob =self.opt.dropout_rate_probs)
#        self.triplet_loss = Lambda(triplet_hinge_loss)

    def __init__(self,opt):
        super(RealNN, self).__init__(opt)

    def build(self):
        rep_m = rep_model(self.opt)
        if self.opt.match_type == 'pointwise':
            rep = []
            for doc in [self.question, self.answer]:
                rep.append(rep_m.get_representation(doc))
            output = self.distance(rep)
#            output =  Cosinse(dropout_keep_prob=self.opt.dropout_rate_probs)(rep) 
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
    from models.BasicModel import BasicModel
    from params import Params
    import numpy as np
    import dataset
    import units
    opt = Params()
    config_file = 'config/local.ini'    # define dataset in the config
    opt.parse_config(config_file)
    reader = dataset.setup(opt)
    opt = dataset.process_embedding(reader,opt)
    
    (train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()
    train_y = np.random.randint(2,size = len(train_y))
    self = BasicModel(opt)
#    model.compile(loss = opt.loss,
#            optimizer = units.getOptimizer(name=opt.optimizer,lr=opt.lr),
#            metrics=['accuracy'])
#    print(model.predict(x = [train_x,train_x]))
#    

