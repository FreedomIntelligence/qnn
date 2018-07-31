import sys
import os
import numpy as np
import codecs
import pandas as pd
sys.path.append('complexnn')

from keras.models import Model, Input, model_from_json, load_model
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout,Activation
from embedding import phase_embedding_layer, amplitude_embedding_layer
from multiply import ComplexMultiply
from data import orthonormalized_word_embeddings,get_lookup_table, batch_gen,data_gen
from mixture import ComplexMixture
from data_reader import *
from superposition import ComplexSuperposition
from measurement import ComplexMeasurement
from keras.preprocessing.sequence import pad_sequences
from projection import Complex1DProjection
from keras.utils import to_categorical
from keras.constraints import unit_norm
from dense import ComplexDense
from utils import GetReal
from keras.initializers import Constant
from params import Params
import matplotlib.pyplot as plt


def qpdnn(lookup_table, max_sequence_length, network_type = 'complex_mixture', nb_classes = 2, measurement_units = 5, random_init = True, embedding_trainable = True, dropout_rate = 0.0, init_mode = 'he', activation = 'sigmoid'):

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    weight_embedding = Embedding(lookup_table.shape[0], 1, trainable = True)(sequence_input)
    weight_embedding = Activation('softmax')(weight_embedding)

    phase_embedding = Dropout(dropout_rate)(phase_embedding_layer(max_sequence_length, lookup_table.shape[0], embedding_dimension, trainable = embedding_trainable)(sequence_input))

    amplitude_embedding = Dropout(dropout_rate)(amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length, trainable = embedding_trainable, random_init = random_init)(sequence_input))

    [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])

    if network_type.lower() == 'complex_mixture':
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, weight_embedding])

    elif network_type.lower() == 'complex_superposition':
        [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, weight_embedding])

    else:
        print('Wrong input network type -- The default mixture network is constructed.')
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, weight_embedding])

    probs = ComplexMeasurement(units = measurement_units)([sentence_embedding_real, sentence_embedding_imag])

    output =  Dense(units = nb_classes)(probs)

    model = Model(sequence_input, output)

    # model.compile(loss='binary_crossentropy',
    #       optimizer=optimizer,
    #       metrics=['accuracy'])
    return model

def qpdnn1(lookup_table, max_sequence_length, network_type = 'complex_mixture', nb_classes = 2, measurement_units = 5, random_init = True, embedding_trainable = True, dropout_rate = 0.0, init_mode = 'he', activation = 'sigmoid'):

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    weight_embedding = Embedding(lookup_table.shape[0], 1, trainable = True)(sequence_input)
    weight_embedding = Activation('softmax')(weight_embedding)

    phase_embedding = Dropout(dropout_rate)(phase_embedding_layer(max_sequence_length, lookup_table.shape[0], embedding_dimension, trainable = embedding_trainable)(sequence_input))

    amplitude_embedding = Dropout(dropout_rate)(amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length, trainable = embedding_trainable, random_init = random_init)(sequence_input))

    [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])

    if network_type.lower() == 'complex_mixture':
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, weight_embedding])

    elif network_type.lower() == 'complex_superposition':
        [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition()([seq_embedding_real, seq_embedding_imag, weight_embedding])

    else:
        print('Wrong input network type -- The default mixture network is constructed.')
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture()([seq_embedding_real, seq_embedding_imag, weight_embedding])

    probs = ComplexMeasurement(units = measurement_units)([sentence_embedding_real, sentence_embedding_imag])

    output =  Dense(units = nb_classes)(probs)

    model = Model(sequence_input, output)

    # model.compile(loss='binary_crossentropy',
    #       optimizer=optimizer,
    #       metrics=['accuracy'])
    return model

def main():
    params = Params()
    params.parse_config('config/config.ini')
    reader = data_reader_initialize(params.dataset_name,params.datasets_dir)
    if(params.wordvec_initialization == 'orthogonalize'):
        embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=True)
    
    elif( (params.wordvec_initialization == 'random') | (params.wordvec_initialization == 'word2vec')):
        embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=False)
    else:
        raise ValueError('The input word initialization approach is invalid!')
    
    train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)
    max_sequence_length = reader.max_sentence_length
    random_init = True
    if not(params.wordvec_initialization == 'random'):
        random_init = False
    
    training_data = train_test_val['train']
    test_data = train_test_val['test']
    validation_data = train_test_val['dev']
    
    train_x, train_y = data_gen(training_data, max_sequence_length)
    test_x, test_y = data_gen(test_data, max_sequence_length)
    val_x, val_y = data_gen(validation_data, max_sequence_length)
    
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    val_y = to_categorical(val_y)



    # print(embedding_params['word2id'])
    lookup_table = get_lookup_table(embedding_params)



    model = qpdnn(lookup_table, max_sequence_length, nb_classes = reader.nb_classes, network_type = params.network_type, random_init = random_init, embedding_trainable = params.embedding_trainable, init_mode = params.init_mode, activation = params.activation)
    model.compile(loss = params.loss,
          optimizer = params.optimizer,
          metrics=['accuracy'])
    model.summary()
    # weights = model.get_weights()
    history = model.fit(x=train_x, y = train_y, batch_size = 1, epochs= params.epochs,validation_data= (test_x, test_y))

    val_acc= history.history['val_acc']
    train_acc = history.history['acc']
    with open("eval.txt") as f:
        model_info = "%.4f test acc,  %4.f train acc , model : %s,  dropout_rate: %.2f, optimizer: %s ,init_mode %s \n " %(max(val_acc),max(train_acc),"mixture" if projection else "superposition",dropout_rate,optimizer,init_mode )
        f.write(model_info)




if __name__ == '__main__':
    main()
