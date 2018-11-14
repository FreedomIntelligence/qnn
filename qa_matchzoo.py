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

from tools.logger import Logger
logger = Logger() 

from models.match.keras import matchzoo as models 
from loss.rankloss import rank_hinge_loss,rank_crossentropy_loss
loss = rank_hinge_loss({"margin":0.1})
loss = rank_crossentropy_loss({"neg_num":5})

if __name__ == '__main__':
    
    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    params.network_type = "anmm.ANMM"
    
    reader = qa.setup(params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    
    
    model.compile(loss = loss,
                optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
                metrics=['accuracy'])
    model.summary()
    
#    generators = [reader.getTrain(iterable=False) for i in range(params.epochs)]
#    q,a,score = reader.getPointWiseSamples()
#    model.fit(x = [q,a],y = score,epochs = 1,batch_size =params.batch_size)
    
    def gen():
        while True:
            for sample in reader.getPointWiseSamples4Keras(iterable = True):
                yield sample
    model.fit_generator(gen(),epochs = 2,steps_per_epoch=1000)

   

#def test_match():
    
    

        
    
    
    
