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

def mse(y_true, y_pred, rel_threshold=0.):
#    s = 0.
#    y_true = _to_list(np.squeeze(y_true).tolist())
#    y_pred = _to_list(np.squeeze(y_pred).tolist())
#    return np.mean(np.square(y_pred - y_true), axis=-1)
    x=K.squeeze(y_pred,axis=1)
    y=K.squeeze(y_true,axis=1)
    return K.mean(K.square(x-y))

def test_matchzoo():
    
    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    params.network_type = "anmm.ANMM"
    
    reader = qa.setup(params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    
    metrics= [map,mrr,ndcg(3),ndcg(5),'accuracy']
    model.compile(loss = params.loss,
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=metrics)
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
    
    
#    model.compile(loss = rank_hinge_loss({'margin':0.2}),
#                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
#                metrics=['accuracy'])
    model.compile(loss = 'mean_squared_error',
            optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
            metrics=['accuracy'])
    q,a,neg  = reader.getTrain(max_sequence_length=reader.max_sequence_length,iterable = False, shuffle = False)
#    model.fit_generator(reader.getPointWiseSamples4Keras(),epochs = 1,steps_per_epoch=1)
    y_pred_pos = model.predict(x = [to_array(q,reader.max_sequence_length),to_array(a, reader.max_sequence_length)])
#    print(y_pred_pos)
    y_pred_neg = model.predict(x = [to_array(q,reader.max_sequence_length),to_array(neg, reader.max_sequence_length)])
#    print(y_pred_neg)
    
    y_pred = np.concatenate([y_pred_pos,y_pred_neg])
#    model.summary()
    
    
    
    
#    generators = [reader.getTrain(iterable=False) for i in range(params.epochs)]
#    [q,a,score] = reader.getPointWiseSamples()
#    model.fit(x = [q,a,a],y = [q,a,q],epochs = 10,batch_size =params.batch_size)
    
#    def gen():
#        while True:
#            for sample in reader.getTrain(iterable = True):
#                yield sample


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
#    test()
    test_match()


# x_input = np.asarray([b])
# y = model.predict(x_input)
# print(y)


