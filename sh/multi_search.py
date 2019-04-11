# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:44:09 2018

@author: wabywang
"""


import sys
import os,time,random
import numpy as np
import codecs
import pandas as pd
sys.path.append('complexnn')
from keras.models import Model, Input, model_from_json, load_model
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten, Dropout
from embedding import phase_embedding_layer, amplitude_embedding_layer
from multiply import ComplexMultiply
from data import orthonormalized_word_embeddings,get_lookup_table, batch_gen,data_gen
from mixture import ComplexMixture
from data_reader import *
from superposition import ComplexSuperposition
from keras.preprocessing.sequence import pad_sequences
from projection import Complex1DProjection
from keras.utils import to_categorical
from keras.constraints import unit_norm
from dense import ComplexDense
from utils import GetReal
from keras.initializers import Constant
from params import Params
from main import run
import matplotlib.pyplot as plt
import argparse




import itertools
import multiprocessing
import GPUUtil


def run_task(zipped_args, params):
    i,(dropout_rate,optimizer,network_type,init_mode,batch_size,activation) = zipped_args

    arg_str=(" ".join([str(ii) for ii in (dropout_rate,optimizer,network_type,init_mode,batch_size,activation)]))
    print ('Run task %s (%d)... \n' % (arg_str, os.getpid()))
#    try:
#        GPUUtil.setCUDA_VISIBLE_DEVICES(num_GPUs=1, verbose=True) != 0
#    except Exception as e:
#        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(i%8))
#        print ('use GPU %d \n' % (int(i%8)))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(i%8))
    print ('use GPU %d \n' % (int(i%8)))

    # model = createModel(dropout_rate,optimizer,init_mode,activation)
    params.dropout_rate = dropout_rate
    params.optimizer = optimizer
    params.network_type = network_type
    params.init_mode = init_mode
    params.activation = activation
    history, evaluation = run(params)
    start=time.time()
    # history = model.fit(x=train_x, y = train_y, batch_size = batch_size, epochs= params.epochs,validation_data= (test_x, test_y),verbose = 0 )

    val_acc= history.history['val_acc']
    train_acc = history.history['acc']


    model_info = "%s:  dropout_rate:%.2f  optimizer:%s init_mode %s batch_size:%d  activation:%s" %(network_type,dropout_rate,optimizer,init_mode,batch_size,activation)
    df = pd.read_csv(params.dataset_name+".csv",index_col=0,sep="\t")
    dataset = params.dataset_name
#    if arg_str not in df:
#        df.loc[arg_str] = pd.Series()
#    if dataset not in df.loc[arg_str]:
    df.loc[model_info,dataset] = max(val_acc)
    df.to_csv(params.dataset_name+".csv",sep="\t")

    print(model_info +" with time :"+ str( time.time()-start)+" ->" +str( max(val_acc) ) )






#    time.sleep(1)
if __name__ == "__main__":
    # import argparse
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu', help = 'please enter the gpu num.')
    args = parser.parse_args()
    print("gpu")
    # gpu = 5
    gpu = int(args.gpu)
    print("gpu : %d" % gpu)

    params = Params()
    params.parse_config('config/config.ini')

    if not os.path.exists(params.dataset_name+".csv"):
        with open(params.dataset_name+".csv","w") as f:
            f.write("argument\t"+params.dataset_name+"\n")
#            f.write("0\n")
            f.close()


#    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dropout_rates = [0.0, 0.1, 0.2,  0.5,  0.8,]
#    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    optimizers = [ 'Adam', 'Nadam']
#    init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform','he']
    network_type = ['complex_superposition','complex_mixture']
    init_modes = ["glorot","he"]
    batch_sizes = [8,32,64,128]
    activations = ["relu","sigmoid","tanh"]



    args = [i for i in enumerate(itertools.product(dropout_rates,optimizers,network_type,init_modes,batch_sizes,activations)) if i[0]%8==gpu]
    for arg in args:
        run_task(arg, params)












