from dense import ComplexDense
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
from keras.initializers import RandomUniform
from keras.constraints import unit_norm
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keras.backend as K
import math
from data import get_lookup_table,data_gen
from data_reader import SSTDataReader
from keras.layers import Embedding
from multiply import ComplexMultiply
from positive_unit_norm import PositiveUnitNorm

def phase_embedding_layer(max_sequence_length, input_dim, embedding_dim = 1,trainable = True):
    embedding_layer = Embedding(input_dim,
                            embedding_dim,
                            embeddings_initializer=RandomUniform(minval=0, maxval=2*math.pi),
                            input_length=max_sequence_length, trainable = trainable)
    return embedding_layer



def amplitude_embedding_layer(embedding_matrix, max_sequence_length, trainable = False, random_init = True):
    embedding_dim = embedding_matrix.shape[0]
    vocabulary_size = embedding_matrix.shape[1]
    if(random_init):
        return(Embedding(vocabulary_size,
                                embedding_dim,
                                embeddings_constraint = unit_norm(axis = 1),
                                input_length=max_sequence_length,
                                trainable=trainable))
    else:
        return(Embedding(vocabulary_size,
                                embedding_dim,
                                weights=[np.transpose(embedding_matrix)],
                                embeddings_constraint = unit_norm(axis = 1),
                                input_length=max_sequence_length,
                                trainable=trainable))






def main():
    path_to_vec = '../glove/glove.6B.100d.txt'
    dir_name = '../'
    reader = SSTDataReader(dir_name,nclasses = 2)
    embedding_params = reader.get_word_embedding(path_to_vec,orthonormalized=False)
    lookup_table = get_lookup_table(embedding_params)
    max_sequence_length = 60


    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    phase_embedding = phase_embedding_layer(max_sequence_length, lookup_table.shape[0])

    amplitude_embedding = amplitude_embedding_layer(np.transpose(lookup_table), max_sequence_length)

    # [embed_seq_real, embed_seq_imag] = ComplexMultiply()([phase_embedding, amplitude_embedding])
    output = phase_embedding(sequence_input)
    model = Model(sequence_input, output)
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()


    train_test_val= reader.create_batch(embedding_params = embedding_params,batch_size = -1)

    training_data = train_test_val['train']
    test_data = train_test_val['test']
    validation_data = train_test_val['dev']


    # for x, y in batch_gen(training_data, max_sequence_length):
    #     model.train_on_batch(x,y)

    train_x, train_y = data_gen(training_data, max_sequence_length)
    test_x, test_y = data_gen(test_data, max_sequence_length)
    val_x, val_y = data_gen(validation_data, max_sequence_length)
    # sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    # path_to_vec = '../glove/glove.6B.100d.txt'
    # embedded_sequences = amplitude_embedding_layer(path_to_vec, 10)

    # output = embedded_sequences(sequence_input)
    # model = Model(sequence_input, output)
    # model.compile(loss='categorical_crossentropy',
    #           optimizer='rmsprop',
    #           metrics=['acc'])

    # model.summary()

    x = train_x

    y = model.predict(x)
    print(y)
    print(y.shape)

    # rng = numpy.random.RandomState(123)

    # Warning: the full dataset is over 40GB. Make sure you have enough RAM!
    # This can take a few minutes to load
    # if in_memory:
    #     print('.. loading train data')
    #     dataset = MusicNet(local_data, complex_=complex_, fourier=fourier,
    #                        stft=stft, rng=rng, fast_load=fast_load)
    #     dataset.load()
    #     print('.. train data loaded')
    #     Xvalid, Yvalid = dataset.eval_set('valid')
    #     Xtest, Ytest = dataset.eval_set('test')
    # else:
    #     raise ValueError

    # print(".. building model")
    # # model = get_shallow_convnet(window_size=4096, channels=2, output_size=84)
    # model = one_hidden_layer_complex_nn(input_size = 300, output_size = 2)
    # model.summary()
    # print(".. parameters: {:03.2f}M".format(model.count_params() / 1000000.))


    # # x =
    # x = np.random.random((1,300))
    # y = to_categorical(np.random.randint(2, size=(1, 1)), num_classes=2)


    # for i in range(700):
    #     model.fit(x,y)

    # print(y)
    # print(model.predict(x))
    # if in_memory:
    #     pass
    #     # do nothing
    # else:
    #     raise ValueError

    # logger = mimir.Logger(
    #     filename='models/log_{}.jsonl.gz'.format(model_name))

    # it = dataset.train_iterator()

    # callbacks = [Validation(Xvalid, Yvalid, 'valid', logger),
    #              Validation(Xtest, Ytest, 'test', logger),
    #              SaveLastModel("./models/", 1, name=model),
    #              Performance(logger),
    #              LearningRateScheduler(schedule)]

    # print('.. start training')
    # model.fit_generator(
    #     it, steps_per_epoch=1000, epochs=epochs,
    #     callbacks=callbacks, workers=1)

if __name__ == '__main__':
    main()
