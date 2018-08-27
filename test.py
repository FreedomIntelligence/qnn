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
import itertools
from loss import *
from units import to_array 
from keras.utils import generic_utils
import argparse

gpu_count = len(units.get_available_gpus())
dir_path,global_logger = units.getLogger()
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
    gpu_count = len(units.get_available_gpus())
    grid_parameters ={
#        "dataset_name":["SST_2"],
#        "wordvec_path":["glove/glove.6B.50d.txt"],#"glove/glove.6B.300d.txt"],"glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt",
#        "loss": ["categorical_crossentropy"],#"mean_squared_error"],,"categorical_hinge"
#        "optimizer":["rmsprop"], #"adagrad","adamax","nadam"],,"adadelta","adam"
#        "batch_size":[16],#,32
#        "activation":["sigmoid"],
#        "amplitude_l2":[0], #0.0000005,0.0000001,
#        "phase_l2":[0.00000005],
#        "dense_l2":[0],#0.0001,0.00001,0],
        "measurement_size" :[5,10,50,100,200,500],#,50100],
        "match_type":['pointwise','pairwise'],
        "network_type": ['real','qdnn','local_mixture']
#        "lr" : [0.1],#,1,0.01
#        "dropout_rate_embedding" : [0.9],#0.5,0.75,0.8,0.9,1],
#        "dropout_rate_probs" : [0.9],#,0.5,0.75,0.8,1]    ,
#        "ablation" : [1],
##        "network_type" : ["ablation"]
    }
    
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    parameters= parameters[::-1]        
    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    for parameter in parameters:
        params.setup(zip(grid_parameters.keys(),parameter))
        reader = qa.setup(params)
        params.reader = reader
        qdnn = models.setup(params)
        model = qdnn.getModel()
        model.summary()
#    model.compile(loss = rank_hinge_loss({'margin':0.2}),
#                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
#                metrics=['accuracy'])
    

        print('############################################################')
        print('checking measurement size:{}'.format(params.measurement_size))
        print('############################################################')
        if params.match_type == 'pointwise':
            model.compile(loss = params.loss,
                    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                    metrics=['accuracy'])
            test_data = reader.getTest(iterable = False)
            test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
            for i in range(params.epochs):
                model.fit_generator(reader.getPointWiseSamples4Keras(),epochs = 1,steps_per_epoch= params.steps_per_epoch)        
                y_pred = model.predict(x = test_data)            
                print(reader.evaluate(y_pred, mode = "test"))
                
        elif params.match_type == 'pairwise':
            model.compile(loss = rank_hinge_loss({'margin':params.margin}),
                    optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                    metrics=['accuracy'])
            
            test_data = reader.getTest(iterable = False)
            test_data.append(test_data[0])
            test_data = [to_array(i,reader.max_sequence_length) for i in test_data]
            progbar = generic_utils.Progbar(params.epochs)
            history = []
            for i in range(params.epochs):
                model.fit_generator(reader.getPairWiseSamples4Keras(),epochs = 1,steps_per_epoch=params.steps_per_epoch,verbose = False)
                
                y_pred = model.predict(x = test_data)
                q = y_pred[0]
                a = y_pred[1]
                score = np.sum((q-a)**2, axis=1)
    #            print(score)
                
                MAP, MRR, PREC_1 = reader.evaluate(score, mode = "test")
                progbar.add(1, values=[('MAP',MAP), ('MRR',MRR), ('PREC_1',PREC_1)])
                history.append((MAP,MRR,PREC_1))

 
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


