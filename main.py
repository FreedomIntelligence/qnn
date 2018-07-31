import sys
import os
import numpy as np
import codecs
import pandas as pd
sys.path.append('complexnn')

from keras.models import Model, Input, model_from_json, load_model
from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
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

def run_complex_embedding_network(lookup_table, max_sequence_length, network_type = 'complex_mixture', nb_classes = 2, random_init = True, embedding_trainable = True, dropout_rate = 0.0, init_mode = 'he', activation = 'sigmoid'):

    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    phase_embedding = Dropout(dropout_rate)(phase_embedding_layer(max_sequence_length, lookup_table.shape[0], embedding_dimension, trainable = embedding_trainable)(sequence_input))


    amplitude_embedding = Dropout(dropout_rate)(amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length, trainable = embedding_trainable, random_init = random_init)(sequence_input))

    weight_embedding = Embedding(lookup_table.shape[0], 1, trainable = False)(sequence_input)
    weight_embedding = Activation('softmax')(weight_embedding)

    [seq_embedding_real, seq_embedding_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])

    if network_type.lower() == 'complex_mixture':
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture(average_weights = True)([seq_embedding_real, seq_embedding_imag, weight_embedding])


    elif network_type.lower() == 'complex_superposition':
        [sentence_embedding_real, sentence_embedding_imag]= ComplexSuperposition(average_weights = True)([seq_embedding_real, seq_embedding_imag, weight_embedding])

    else:
        print('Wrong input network type -- The default mixture network is constructed.')
        [sentence_embedding_real, sentence_embedding_imag]= ComplexMixture(average_weights = True)([seq_embedding_real, seq_embedding_imag, weight_embedding])

    sentence_embedding_real = Flatten()(sentence_embedding_real)
    sentence_embedding_imag = Flatten()(sentence_embedding_imag)
    # output = Complex1DProjection(dimension = embedding_dimension)([sentence_embedding_real, sentence_embedding_imag])
    predictions = ComplexDense(units = nb_classes, activation= activation, bias_initializer=Constant(value=-1), init_criterion = init_mode)([sentence_embedding_real, sentence_embedding_imag])

    output = GetReal()(predictions)

    model = Model(sequence_input, output)

    # model.compile(loss='binary_crossentropy',
    #       optimizer=optimizer,
    #       metrics=['accuracy'])
    return model


def run_real_embedding_network(lookup_table, max_sequence_length, nb_classes = 2, random_init = True, embedding_trainable = True, dropout_rate = 0.0, activation = 'sigmoid'):
    embedding_dimension = lookup_table.shape[1]
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    if(random_init):
        embedding = Dropout(dropout_rate)(Embedding(trainable=embedding_trainable, input_dim=lookup_table.shape[0],output_dim=lookup_table.shape[1], weights=[lookup_table],embeddings_constraint = unit_norm(axis = 1))(sequence_input))
    else:
        embedding = Embedding(trainable=embedding_trainable, input_dim=lookup_table.shape[0],output_dim=lookup_table.shape[1],embeddings_constraint = unit_norm(axis = 1))(sequence_input)
    representation =GlobalAveragePooling1D()(embedding)
    output = Dense(nb_classes, activation=activation)(representation)
    model = Model(sequence_input, output)

    return model

def save_model(model, model_dir):
    if not (os.path.exists(model_dir)):
        os.mkdir(model_dir)

    model.save_weights(os.path.join(model_dir,'weight'))
    json_string = model.to_json()
    data_out = codecs.open(os.path.join(model_dir,'model_structure.json'),'w')
    data_out.write(json_string)
    data_out.close()

def load_model(model_dir, params):
    json_file = open(os.path.join(model_dir,'model_structure.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    custom_layers = {'ComplexMultiply': ComplexMultiply, 'ComplexMixture': ComplexMixture, 'ComplexDense': ComplexDense,'GetReal': GetReal}

    model = model_from_json(loaded_model_json, custom_objects=
        custom_layers)
    model.compile(loss = params.loss,
          optimizer = params.optimizer,
          metrics=['accuracy'])
    model.load_weights(os.path.join(model_dir,'weight'))
    return(model)

def run(params):
    # datasets_dir, dataset_name, wordvec_initialization ='random', wordvec_path = None, loss = 'binary_crossentropy', optimizer = 'rmsprop', batch_size = 16, epochs= 4

    reader = data_reader_initialize(params.dataset_name,params.datasets_dir)

    if(params.wordvec_initialization == 'orthogonalize'):
        embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=True)

    elif( (params.wordvec_initialization == 'random') | (params.wordvec_initialization == 'word2vec')):
        embedding_params = reader.get_word_embedding(params.wordvec_path,orthonormalized=False)
    else:
        raise ValueError('The input word initialization approach is invalid!')

    # print(embedding_params['word2id'])
    lookup_table = get_lookup_table(embedding_params)

    max_sequence_length = reader.max_sentence_length
    random_init = True
    if not(params.wordvec_initialization == 'random'):
        random_init = False

    if params.network_type == 'real':
        model = run_real_embedding_network(lookup_table, max_sequence_length, nb_classes = reader.nb_classes, random_init = random_init, dropout_rate = param.dropout_rate, embedding_trainable = params.embedding_trainable, activation = params.activation)
    else:
        model = run_complex_embedding_network(lookup_table, max_sequence_length, nb_classes = reader.nb_classes, network_type = params.network_type, random_init = random_init, embedding_trainable = params.embedding_trainable, init_mode = params.init_mode, activation = params.activation)

    model.compile(loss = params.loss,
          optimizer = params.optimizer,
          metrics=['accuracy'])

    model.summary()
    weights = model.get_weights()


    train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)

    training_data = train_test_val['train']
    test_data = train_test_val['test']
    validation_data = train_test_val['dev']


    # for x, y in batch_gen(training_data, max_sequence_length):
    #     model.train_on_batch(x,y)

    train_x, train_y = data_gen(training_data, max_sequence_length)
    test_x, test_y = data_gen(test_data, max_sequence_length)
    val_x, val_y = data_gen(validation_data, max_sequence_length)
    # assert len(train_x) == 67349
    # assert len(test_x) == 1821
    # assert len(val_x) == 872

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    val_y = to_categorical(val_y)

    history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))

    evaluation = model.evaluate(x = test_x, y = test_y)
    return (history, evaluation)

def evaluate(history, evaluation, eval_dir):

    val_acc= history.history['val_acc']
    train_acc = history.history['acc']

    if not os.path.exists(params.eval_dir):
        os.mkdir(params.eval_dir)

    learning_curve_path = os.path.join(params.eval_dir,'learning_curve')
    epoch_indexes = [x+1 for x in range(len(val_acc))]
    line_1, = plt.plot(epoch_indexes, val_acc)
    line_2, = plt.plot(epoch_indexes, train_acc)
    # plt.axis([0, 6, 0, 20])

    plt.legend([line_1, line_2], ['test_acc', 'train_acc'])
    fig = plt.gcf()
    fig.savefig(learning_curve_path, dpi=fig.dpi)

    evaluation = model.evaluate(x = test_x, y = test_y)
    eval_file_path = os.path.join(params.eval_dir,'eval.txt')

    with open(eval_file_path,'w') as eval_file:
        eval_file.write('acc: {}, loss: {}'.format(evaluation[1], evaluation[0]))


    embedding_dir = os.path.join(params.eval_dir,'embedding')
    if not(os.path.exists(embedding_dir)):
        os.mkdir(embedding_dir)
    np.save(os.path.join(embedding_dir,'phase_embedding'), model.get_weights()[0])
    np.save(os.path.join(embedding_dir,'amplitude_embedding'), model.get_weights()[1])
    np.save(os.path.join(embedding_dir,'word2id'), embedding_params['word2id'])
    save_model(model, os.path.join(params.eval_dir,'model'))


    experiment_results_path = 'eval/experiment_result.xlsx'
    xls_file = pd.ExcelFile(experiment_results_path)

    df1 = xls_file.parse('Sheet1')
    l = {'complex_mixture':0,'complex_superposition':1,'real':2}
    df1.ix[l[params.network_type],params.dataset_name] = max(val_acc)
    df1.to_excel(experiment_results_path)
    # model_2 = load_model(os.path.join(params.eval_dir,'model'), params)
    # print(model_2.evaluate(x = test_x, y = test_y))
    # print(evaluation)




if __name__ == '__main__':
    params = Params()
    params.parse_config('config/config.ini')
    print(params.network_type)
    # params.parseArgs()
    history, evaluation = run(params)
    evaluate(history, evaluation, params.eval_dir)


    #################################################################




