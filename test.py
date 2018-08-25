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
    

def test_match():
    from models.match import keras as models
    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    
    reader = qa.setup(params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    metrics= [map,mrr,ndcg(3),ndcg(5)]
    
#    model.compile(loss = rank_hinge_loss({'margin':0.2}),
#                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
#                metrics=['accuracy'])
    
    test_data = reader.getTest(iterable = False)
    test_data.append(test_data[0])
    test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
    if params.match_type == 'pointwise':
        model.compile(loss = params.loss,
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=['accuracy'])
        
        for i in range(params.epochs):
            model.fit_generator(reader.getPointWiseSamples4Keras(),epochs = 1,steps_per_epoch=1000)        
            y_pred = model.predict(x = test_data)            
            print(reader.evaluate(y_pred, mode = "test"))
            
    elif params.match_type == 'pairwise':
        model.compile(loss = rank_hinge_loss({'margin':params.margin}),
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=['accuracy'])
        
        for i in range(params.epochs):
            model.fit_generator(reader.getPairWiseSamples4Keras(),epochs = 1,steps_per_epoch=50,verbose = False)

            y_pred = model.predict(x = test_data)
            q = y_pred[0]
            a = y_pred[1]
            score = np.sum((q-a)**2, axis=1)
#            print(score)
            print(reader.evaluate(score, mode = "test"))
            


def test():
    import models.representation as models
    params = Params()
    config_file = 'config/local.ini'    # define dataset in the config
    params.parse_config(config_file)
    import dataset
    reader = dataset.setup(params)
    params = dataset.process_embedding(reader,params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    
    model.compile(loss = params.loss,
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=['accuracy'])
    model.summary()
    (train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()
    history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))

    evaluation = model.evaluate(x = val_x, y = val_y)
    
if __name__ == '__main__':
    test()
#    test_match()


# x_input = np.asarray([b])
# y = model.predict(x_input)
# print(y)


