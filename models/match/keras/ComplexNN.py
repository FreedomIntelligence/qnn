
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
from models.representation.ComplexNN import ComplexNN as rep_model

class ComplexNN(BasicModel):
    
    def initialize(self):
        self.question = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.neg_answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.distance = Lambda(l2_distance)
        
    def __init__(self,opt):
        super(ComplexNN, self).__init__(opt) 
        
    def build(self):
        rep_m = rep_model(self.opt)
        if self.opt.match_type == 'pointwise':
            rep = []
            for doc in [self.question, self.answer]:
                # Take the real part of the output
                rep.append(rep_m.get_representation(doc)[0])
            output = self.distance(rep)
            model = Model([self.question, self.answer], output)
        elif self.opt.match_type == 'pairwise':
            rep = []
            for doc in [self.question, self.answer, self.neg_answer]:
                rep.append(rep_m.get_representation(doc)[0])
            output = rep
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

