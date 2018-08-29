# -*- coding: utf-8 -*-
import keras
from keras.layers import Input, Dense, Activation, Lambda
import numpy as np
from keras import regularizers
from keras.models import Model
import sys
from params import Params
from dataset import qa
import keras.backend as K
import units

from tools.evaluationKeras import map,mrr,ndcg

from loss import *
from units import to_array 
def identity_loss(y_true, y_pred):

    return K.mean(y_pred)


def percision_bacth(y_true, y_pred):
    return K.mean(K.cast(K.equal(y_pred,0),"float32"))

def test_matchzoo():
    
    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    params.network_type = "anmm.ANMM"
    
    reader = qa.setup(params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    
    
    model.compile(loss = params.loss,
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=['accuracy'])
    model.summary()
    
#    generators = [reader.getTrain(iterable=False) for i in range(params.epochs)]
#    q,a,score = reader.getPointWiseSamples()
#    model.fit(x = [q,a],y = score,epochs = 1,batch_size =params.batch_size)
    
    def gen():
        while True:
            for sample in reader.getPointWiseSamples(iterable = True):
                yield sample
    model.fit_generator(gen(),epochs = 2,steps_per_epoch=1000)
    
if __name__ == '__main__':
#def test_match():
    from models.match import keras as models
    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    
    reader = qa.setup(params)
    qdnn = models.setup(params)
    model = qdnn.getModel()

    
#    model.compile(loss = rank_hinge_loss({'margin':0.2}),
#                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
#                metrics=['accuracy'])
    
    test_data = reader.getTest(iterable = False)
#    test_data.append(test_data[0])
    
    if params.match_type == 'pointwise':
        
#        test_data = [to_array(i,reader.max_sequence_length) for i in test_data[:2]]
        
        model.compile(loss = params.loss,
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=['accuracy'])
        
        for i in range(params.epochs):
            model.fit_generator(reader.getPointWiseSamples4Keras(),epochs = 1,steps_per_epoch=1000)        
            y_pred = model.predict(x = test_data)            
            print(reader.evaluate(y_pred, mode = "test"))
            
    elif params.match_type == 'pairwise':
        test_data.append(test_data[0])
        test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
        model.compile(loss = identity_loss,
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=[percision_bacth],
                loss_weights=[0.0, 1.0,0.0])
        
        for i in range(params.epochs):
            model.fit_generator(reader.getPairWiseSamples4Keras(),epochs = 1,steps_per_epoch=22,verbose = True)

            y_pred = model.predict(x = test_data)

            score = y_pred[0]
#            print(score)
            print(reader.evaluate(score, mode = "test"))
            

    



