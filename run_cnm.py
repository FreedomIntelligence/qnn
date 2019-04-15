# -*- coding: utf-8 -*-
from params import Params
from dataset import qa
import keras.backend as K
import pandas as pd
from layers.loss import *
from layers.loss.metrics import precision_batch
from tools.units import to_array, getOptimizer,parse_grid_parameters
import argparse
import itertools
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import os
import random
from models import match as models   
from tools.evaluation import matching_score

def run(params,reader):
    test_data = reader.getTest(iterable = False, mode = 'test')
    dev_data = reader.getTest(iterable = False, mode = 'dev')
    qdnn = models.setup(params)
    model = qdnn.getModel()

    performance = []
    data_generator = None
    if 'onehot' not in params.__dict__:
        params.onehot = 0
    if params.match_type == 'pointwise':
        test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
        dev_data = [to_array(i,reader.max_sequence_length) for i in dev_data]
        if params.onehot:
            loss_type,metric_type = ("categorical_hinge","acc") 
        else:
            loss_type,metric_type = ("mean_squared_error","mean_squared_error")
            
        model.compile(loss =loss_type, #""
                optimizer = getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=[metric_type])
        data_generator = reader.get_pointwise_samples(onehot = params.onehot)
#            if "unbalance" in  params.__dict__ and params.unbalance:
#                model.fit_generator(reader.getPointWiseSamples4Keras(onehot = params.onehot,unbalance=params.unbalance),epochs = 1,steps_per_epoch=int(len(reader.datas["train"])/reader.batch_size),verbose = True)        
#            else:
#                model.fit_generator(reader.getPointWiseSamples4Keras(onehot = params.onehot),epochs = 1,steps_per_epoch=len(reader.datas["train"]["question"].unique())/reader.batch_size,verbose = True)        
    
    elif params.match_type == 'pairwise':
        test_data.append(test_data[0])
        test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
        dev_data.append(dev_data[0])
        dev_data = [to_array(i,reader.max_sequence_length) for i in dev_data]
        model.compile(loss = identity_loss,
                optimizer = getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=[precision_batch],
                loss_weights=[0.0, 1.0,0.0])
        data_generator = reader.get_pairwise_samples()
          
    
    print('Training the network:')
    for i in range(params.epochs):
        model.fit_generator(data_generator,epochs = 1,steps_per_epoch=int(len(reader.datas["train"]["question"].unique())/reader.batch_size),verbose = True)          
        
        print('Validation Performance:')
        y_pred = model.predict(x = dev_data) 
        score = matching_score(y_pred, params.onehot, params.match_type)
        dev_metric = reader.evaluate(score, mode = "dev")
        print(dev_metric)
        
         
        print('Test Performance:')
        y_pred = model.predict(x = test_data) 
        score = matching_score(y_pred, params.onehot, params.match_type)    
        test_metric = reader.evaluate(score, mode = "test")
        print(test_metric)
        performance.append(dev_metric+test_metric)
        
    print('Done.')
    return performance
            



def write_in_file(file_writer,performance):
    df=pd.DataFrame([list(performance)],columns=["map_dev","mrr_dev","p1_dev","map_test","mrr_test","p1_test"])
    file_writer.write(params.to_string()+'\n')
    file_writer.write(str(df[df.map_dev==df.map_dev.max()])+'\n')
    file_writer.write('_________________________\n\n\n')
    file_writer.flush()
    
    
if __name__ == '__main__':

    
    params = Params()
    parser = argparse.ArgumentParser(description='Running the Complex-valued Network for Matching.')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/qalocal_pair_trec.ini')
    parser.add_argument('-grid_search',action = 'store', dest = 'grid_search',type = bool, help = 'please enter yes for grid search of parameters.', default=False)
    parser.add_argument('-grid_param_file',action = 'store', dest = 'config_grid',help = 'please enter the file storing parameters for ablation', default = 'config/grid_parameters.ini')
    args = parser.parse_args()
    params.parse_config(args.config)
    
#   Reproducibility Setting
    seed(params.seed)
    set_random_seed(params.seed)
    random.seed(params.seed)
    
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    file_writer = open(params.output_file,'w')
    if args.grid_search:
        print('Grid Search Begins.')
        grid_parameters = parse_grid_parameters(args.config_grid)
        print(grid_parameters)
        parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]

        for parameter in parameters:
            params.setup(zip(grid_parameters.keys(),parameter))      
            reader = qa.setup(params)
            performance = run(params, reader)
            write_in_file(file_writer,performance)
            K.clear_session()
            
    else:
        reader = qa.setup(params)
        performance = run(params, reader)
        write_in_file(file_writer,performance)
        K.clear_session()

   
            
                
    
        
    
    
    
