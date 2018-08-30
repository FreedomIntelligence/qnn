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
import pandas as pd

import random
import tensorflow as tf
random.seed(49999)
np.random.seed(49999)
tf.set_random_seed(49999)


from loss import *
from units import to_array 
def identity_loss(y_true, y_pred):

    return K.mean(y_pred)


def pointwise_loss(y_true, y_pred):
    
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
    
    
    
    grid_parameters ={
#        "dataset_name":["MR","TREC","SST_2","SST_5","MPQA","SUBJ","CR"],
#        "wordvec_path":["glove/glove.6B.50d.txt"],#"glove/glove.6B.300d.txt"],"glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt",
#        "loss": ["categorical_crossentropy"],#"mean_squared_error"],,"categorical_hinge"
#        "optimizer":["rmsprop"], #"adagrad","adamax","nadam"],,"adadelta","adam"
#        "batch_size":[16],#,32
#        "activation":["sigmoid"],
#        "amplitude_l2":[0.0000005],
#        "phase_l2":[0.00000005],
#        "dense_l2":[0],#0.0001,0.00001,0],
#        "measurement_size" :[100,200],#,50100],
#        "ngram_value":["1,2,3","2,3,4","1,3,4","3,4"],
#        "margin":[0.1,0.2],
#        "lr" : [0.5,0.2],#,1,0.01
#        "dropout_rate_embedding" : [0.9],#0.5,0.75,0.8,0.9,1],
#        "dropout_rate_probs" : [0.8,0.9]#,0.5,0.75,0.8,1]   
#            "ngram_value" : [3]
    }
    import argparse
    import itertools

    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)

    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    reader = qa.setup(params)
    test_data = reader.getTest(iterable = False)
    for parameter in parameters:
#        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(),parameter))
        from models.match import keras as models      
    
        
        qdnn = models.setup(params)
        model = qdnn.getModel()
    
        
    #    model.compile(loss = rank_hinge_loss({'margin':0.2}),
    #                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
    #                metrics=['accuracy'])
        
        
    #    test_data.append(test_data[0])
        evaluations=[]
        if params.match_type == 'pointwise':
            
            test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
            
            model.compile(loss ="mean_squared_error",
                    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                    metrics=['mean_squared_error'])
            
            for i in range(params.epochs):
                model.fit_generator(reader.getPointWiseSamples4Keras(),epochs = 1,steps_per_epoch=64,verbose = False)        
                y_pred = model.predict(x = test_data)            
                metric=reader.evaluate(y_pred, mode = "test")
                print(metric)
                evaluations.append(metric)
                
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
                metric = reader.evaluate(score, mode = "test")
                print(metric)
                evaluations.append(metric)
        print(parameter)
        df=pd.DataFrame(evaluations,columns=["map","mrr","p1"])
        print(df.max())
        print("_____________")
            
                
    
        
    
    
    
