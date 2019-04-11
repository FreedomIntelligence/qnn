# -*- coding: utf-8 -*-

import sys
import os
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
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier  
from sklearn.grid_search import GridSearchCV  


def createModel(dropout_rate=0.5):
#    projection= True,max_sequence_length=56,nb_classes=2,dropout_rate=0.5,embedding_trainable=True,random_init=False
    projection= True
    max_sequence_length=56
    nb_classes=2
    dropout_rate=0.5
    embedding_trainable=True
    random_init=False
    print("create model : " +dropout_rate )

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    phase_embedding =Dropout(dropout_rate) (phase_embedding_layer(max_sequence_length, lookup_table.shape[0], embedding_dimension, trainable = embedding_trainable)(sequence_input))

    
    amplitude_embedding = Dropout(dropout_rate)(amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length, trainable = embedding_trainable, random_init = random_init)(sequence_input))
    
    
    [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])

    if(projection):
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag])
        sentence_embedding_real = Flatten()(sentence_embedding_real)
        sentence_embedding_imag = Flatten()(sentence_embedding_imag)    
        
    else:
        [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag])

    # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
    predictions = ComplexDense(units = nb_classes, activation='sigmoid', bias_initializer=Constant(value=-1))([sentence_embedding_real, sentence_embedding_imag])

    output = GetReal()(predictions)
    model = Model(sequence_input, output)
    model.compile(loss = params.loss,
          optimizer = params.optimizer,
          metrics=['accuracy'])
    
    return model

def gridsearch(params):  

    max_sequence_length = reader.max_sentence_length
    random_init = True
    if not(params.wordvec_initialization == 'random'):
        random_init = False

    train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)

    training_data = train_test_val['train']
    test_data = train_test_val['test']
    validation_data = train_test_val['dev']


    # for x, y in batch_gen(training_data, max_sequence_length):
    #     model.train_on_batch(x,y)

    train_x, train_y = data_gen(training_data, max_sequence_length)
    test_x, test_y = data_gen(test_data, max_sequence_length)
    val_x, val_y = data_gen(validation_data, max_sequence_length)


    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    val_y = to_categorical(val_y)
    

    

    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
    dropout_rate = [ 0.5, 0.9]  
    param_grid = dict(dropout_rate=dropout_rate)
    
#    ,validation_data= (test_x, test_y)
    model = KerasClassifier(build_fn=createModel, nb_epoch= params.batch_size, batch_size= params.batch_size) #verbose=0
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)#n_jobs=-1   for multi process, but it does not work for GPU
    grid_result = grid.fit(train_x, train_y)  
    # summarize results  
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))  
    for params, mean_score, scores in grid_result.grid_scores_:  
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))  
    
    
    
    experiment_results_path = 'eval/experiment_result.xlsx'
    xls_file = pd.ExcelFile(experiment_results_path)

    df1 = xls_file.parse('Sheet1')
    l = {'complex_mixture':0,'complex_superposition':1,'real':2}
    df1.ix[l[params.network_type],params.dataset_name] = max(grid_result.best_score_)
    df1.to_excel(experiment_results_path)


if __name__ == '__main__':
    params = Params()
    params.parse_config('config/waby.ini')
    
    
    reader = data_reader_initialize(params.dataset_name,params.datasets_dir)

    if(params.wordvec_initialization == 'orthogonalize'):
        embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=True)

    elif( (params.wordvec_initialization == 'random') | (params.wordvec_initialization == 'word2vec')):
        embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=False)
    else:
        raise ValueError('The input word initialization approach is invalid!')

    # print(embedding_params['word2id'])
    lookup_table = get_lookup_table(embedding_params)
    
    gridsearch(params)







