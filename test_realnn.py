# -*- coding: utf-8 -*-
from params import Params
#from models import representation as models

#
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import dataset
import units
from units import to_array, batch_softmax_with_first_item
#from tools.save import save_experiment
import itertools
import argparse
import keras.backend as K
import numpy as np
import preprocess.embedding
from keras.models import Model
import tensorflow as tf
from loss import *
import pandas as pd

from keras.losses import *
from loss.pairwise_loss import *
from loss.triplet_loss import *
import loss.pairwise_loss
import loss.triplet_loss
import models
#from distutils.util import strtobool

gpu_count = len(units.get_available_gpus())
dir_path,global_logger = units.getLogger()

from tools.logger import Logger
logger = Logger()     
import os



def myzip(train_x,train_x_mask):
    assert train_x.shape == train_x_mask.shape
    results=[]
    for i in range(len(train_x)):
        results.append((train_x[i],train_x_mask[i]))
    return results


def run(params):
    evaluation=[]
#    params=dataset.classification.process_embedding(reader,params)    
    qdnn = models.setup(params)
    model = qdnn.getModel()
    model.summary()
    if hasattr(loss.pairwise_loss, params.loss): 
        loss_func = getattr(loss.pairwise_loss, params.loss)
    else:
        loss_func = params.loss
    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr)
#    
    
#    test_data = [to_array(i,params.max_sequence_length) for i in test_data]
    if hasattr(loss.pairwise_loss, params.metric_type):
        metric_func = getattr(loss.pairwise_loss, params.metric_type)
    else:
        metric_func = params.metric_type
    
    model.compile(loss = loss_func, #""
                      optimizer = optimizer,
                      metrics=[metric_func])
    # pairwise:
    # loss = identity_loss
    # metric = precision_batch

    # pointwise:
    # loss = categorical_hinge or mean_squared_error
    # metric = acc or mean_squared_error
    
    # classification:
    # loss = mean_squared_error
    # matrix = acc
      
    if params.dataset_type == 'qa':
        test_data = params.reader.get_test(iterable = False)
#        from models.match import keras as models   
        for i in range(params.epochs):
            model.fit_generator(params.reader.batch_gen(params.reader.get_train(iterable = True)),epochs = 1,steps_per_epoch=int(len(reader.datas["train"])/reader.batch_size),verbose = True)        
            y_pred = model.predict(x = test_data) 
            score = batch_softmax_with_first_item(y_pred)[:,1]  if params.onehot else y_pred
                
            metric = params.reader.evaluate(score, mode = "test")
            evaluation.append(metric)
            print(metric)
            logger.info(metric)
        df=pd.DataFrame(evaluation,columns=["map","mrr","p1"]) 

            
    elif params.dataset_type == 'classification':
        train_x,train_y = params.reader.get_train(iterable = False)
        test_x, test_y = params.reader.get_test(iterable = False)
        val_x,val_y = params.reader.get_val(iterable = False)
        
        history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))
        
        metric = model.evaluate(x = val_x, y = val_y)   # !!!!!! change the order to val and test
        
        evaluation.append(metric)
        logger.info(metric)
        print(history)
        print(metric)

        df=pd.DataFrame(evaluation,columns=["map","mrr","p1"])  
        
    logger.info("\n".join([params.to_string(),"score: "+str(df.max().to_dict())]))

    K.clear_session()


if __name__=="__main__":

 # import argparse
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()      
    params = Params()
    config_file = 'config/config_realnn.ini'    # define dataset in the config
    params.parse_config(config_file)    
    
    reader = dataset.setup(params)
    params.reader = reader
        
    run(params)

