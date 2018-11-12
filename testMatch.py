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
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
tf.set_random_seed(49999)

def batch_softmax_with_first_item(x):
    x_exp = np.exp(x)
    x_sum = np.repeat(np.expand_dims(np.sum(x_exp, axis=1),1), x.shape[1], axis=1)
    return x_exp / x_sum
    

from loss import *
from units import to_array 
def identity_loss(y_true, y_pred):

    return K.mean(y_pred)


def pointwise_loss(y_true, y_pred):
    
    return K.mean(y_pred)


def hinge(y_true, y_pred):
    y_pred= y_pred*2-1
    return K.mean(K.maximum(0.5 - y_true * y_pred, 0.), axis=-1)


def batch_pairwise_oss(y_true, y_pred):
    pos = K.mean(y_true * y_pred, axis=-1)
    neg = K.mean((1. - y_true) * y_pred, axis=-1)
    return K.maximum(neg - pos + 0.1, 0.)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(neg - pos + 1., 0.)



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
        "optimizer":["rmsprop"], #"adagrad","adamax","nadam"],,"adadelta","adam"
        "batch_size":[16],#,32
#        "activation":["sigmoid"],
#        "amplitude_l2":[0.0000005],
#        "phase_l2":[0.00000005],
#        "dense_l2":[0],#0.0001,0.00001,0],
        "measurement_size" :[300],#,50100],
#        "ngram_value":["1,2,3","2,3,4","1,3,4"],
#        "margin":[0.1,0.2],
        "lr" : [0.5],#,1,0.01
#        "dropout_rate_embedding" : [0.9],#0.5,0.75,0.8,0.9,1],
#        "dropout_rate_probs" : [0.8,0.9]#,0.5,0.75,0.8,1]   
#            "ngram_value" : [3]
#        "max_len":[100],
#        "one_hot": [1],
#        "dataset_name": ["wiki","trec"],
#        "pooling_type": ["max","average","none"],
        "distance_type":[6],
        "train_verbose":[0],
        "remove_punctuation": [0],
        "stem" : [0],
        "remove_stopwords" : [0],        
        "max_len":[100],
        "one_hot": [0],
    }
    import argparse
    import itertools

    params = Params()
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
    args = parser.parse_args()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    params.parse_config(args.config)
    file_writer = open(params.output_file,'w')
    for parameter in parameters:
#        old_dataset = params.dataset_name
#        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(),parameter))
#        if old_dataset != params.dataset_name:   # batch_size
#            print("switch %s to %s"%(old_dataset,params.dataset_name))
#            reader=dataset.setup(params)
#            params.reader = reader
        from models.match import keras as models      
        reader = qa.setup(params)
        test_data = reader.getTest(iterable = False)
        print(params.batch_size)
        qdnn = models.setup(params)
        model = qdnn.getModel()
    
        
    #    model.compile(loss = rank_hinge_loss({'margin':0.2}),
    #                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
    #                metrics=['accuracy'])
        
        
    #    test_data.append(test_data[0])
        print(parameter)
        evaluations=[]
        if params.match_type == 'pointwise':
            if params.onehot:
                params.lr = 10 *params.lr
            test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
            loss_type,metric_type = ("categorical_hinge","acc") if params.onehot else ("mean_squared_error","mean_squared_error")
            model.compile(loss =loss_type, #""
                    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                    metrics=[metric_type])
            for i in range(params.epochs):
                if "unbalance" in  params.__dict__ and params.unbalance:
                    model.fit_generator(reader.getPointWiseSamples4Keras(onehot = params.onehot,unbalance=params.unbalance),epochs = 1,steps_per_epoch=int(len(reader.datas["train"])/reader.batch_size),verbose = True)        
                else:
                    model.fit_generator(reader.getPointWiseSamples4Keras(onehot = params.onehot),epochs = 1,steps_per_epoch=len(reader.datas["train"]["question"].unique())/reader.batch_size,verbose = True)        
                y_pred = model.predict(x = test_data) 
                score =batch_softmax_with_first_item(y_pred)[:,1]  if params.onehot else y_pred
                
                metric = reader.evaluate(score, mode = "test")
                evaluations.append(metric)
                print(metric)
            df=pd.DataFrame(evaluations,columns=["map","mrr","p1"])
            file_writer.write(params.to_string()+'\n')
            file_writer.write(str(df.max())+'\n')
            file_writer.write('_________________________\n\n\n')
        #        print("_____________")
            K.clear_session()
        
              
        elif params.match_type == 'pairwise':
            test_data.append(test_data[0])
            test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
#            model.summary()
            model.compile(loss = identity_loss,
                    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                    metrics=[percision_bacth],
                    loss_weights=[0.0, 1.0,0.0])
            
            for i in range(params.epochs):
                model.fit_generator(reader.getPairWiseSamples4Keras(),epochs = 1,steps_per_epoch=int(len(reader.datas["train"]["question"].unique())/reader.batch_size),verbose = True)
#            for i in range(1):
#                model.fit_generator(reader.getPairWiseSamples4Keras(),epochs = 1,steps_per_epoch=1,verbose = True)

                y_pred = model.predict(x = test_data)
                score = y_pred[0]
#                score = batch_softmax_with_first_item(y_pred[0])[:,1]  if params.onehot else y_pred[0][:,1]
                metric = reader.evaluate(score, mode = "test")
                evaluations.append(metric)
                print(metric)
            df=pd.DataFrame(evaluations,columns=["map","mrr","p1"])
            file_writer.write(params.to_string()+'\n')
            file_writer.write(str(df.max())+'\n')
            file_writer.write('_________________________\n\n\n')
        #        print("_____________")
            K.clear_session()
            
                
    
        
    
    
    
