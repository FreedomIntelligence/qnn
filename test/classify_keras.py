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

from tools.logger import Logger
logger = Logger()

 




def test_rep():
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
#    print(train_x.shape,train_y.shape)
    history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))

    evaluation = model.evaluate(x = val_x, y = val_y)
    

if __name__ == '__main__':
    grid_parameters ={
#        "dataset_name":["MR","TREC","SST_2","SST_5","MPQA","SUBJ","CR"],
#        "wordvec_path":["glove/glove.6B.50d.txt"],#"glove/glove.6B.300d.txt"],"glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt",
#        "loss": ["categorical_crossentropy"],#"mean_squared_error"],,"categorical_hinge"
#        "optimizer":["rmsprop"], #"adagrad","adamax","nadam"],,"adadelta","adam"
        "batch_size":[16],#,32
#        "activation":["sigmoid"],
        "amplitude_l2":[0.0000005,0.0000001,0],
        "phase_l2":[0.00000005,0.0000005],
#        "dense_l2":[0],#0.0001,0.00001,0],
#        "measurement_size" :[1400,1600,1800,2000],#,50100],
        "lr" : [0.5],#,1,0.01
#        "dropout_rate_embedding" : [0.9],#0.5,0.75,0.8,0.9,1],
        "dropout_rate_probs" : [0.8]#,0.5,0.75,0.8,1]   
#            "ngram_value" : [3]
    }
    import argparse
    import itertools

    params = Params()
    config_file = 'config/local.ini'    # define dataset in the config
    params.parse_config(config_file)

    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    for parameter in parameters:
#        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(),parameter))
        
        import models.representation as models
        import dataset
        reader = dataset.setup(params)
        params = dataset.process_embedding(reader,params)
        qdnn = models.setup(params)
        model = qdnn.getModel()
        
        model.compile(loss = params.loss,
                    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                    metrics=['accuracy'])
#        model.summary()
        (train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()
        history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y)
        ,verbose=False,callbacks=[logger.getCSVLogger()])#,verbose=False
        logger.info(parameter)
        logger.info(max(history.history["val_acc"]))
        evaluation = model.evaluate(x = val_x, y = val_y)
        K.clear_session()
#    test_match()


# x_input = np.asarray([b])
# y = model.predict(x_input)
# print(y)


