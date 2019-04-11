from utils import *
from dense import ComplexDense
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf
import sys
import os
import keras.backend as K
import math

class ComplexNaive(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(ComplexNaive, self).__init__(**kwargs)


    def build(self, input_shape):

        # Create a trainable weight variable for this layer.

        if len(input_shape) != 2:
             raise ValueError('This layer should be called '
                              'on two inputs. '
                              'Got ' + str(len(input_shape)) + ' inputs.')


        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(ComplexNaive, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on only 2 input.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        output_1 = inputs[0]
        output_2 = inputs[1]
        return [output_1,output_2]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))

        return input_shape


def main():

    input_1 = Input(shape=(10,2), dtype='float')
    input_2 = Input(shape=(10,2), dtype='float')

    [output_1,output_2] = ComplexNaive()([input_1, input_2])

    model = Model([input_1,input_2], [output_1,output_2])

    model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['acc'])
    model.summary()


    # # for i in range(1000):
    x = np.random.random((1,10,2))
    x_2 = x

    output = model.predict([x,x_2])
    print(output)
    # # y = K.sum(K.square(x), axis=None, keepdims = False)
    # x = x/np.linalg.norm(x, ord = 2, axis = (1,2))
    # # print(np.linalg.norm(x[0], ord = 2))
    #     # # print(np.linalg.norm(x))
    # y = model.predict(x)
    # model.fit(x,y)
    # for i in range(100):
    #     x = np.random.random((1,10,2))
    #     x = x/np.linalg.norm(x, ord = 2, axis = (1,2))
    #     y = model.predict(x)
    #     print(y)


if __name__ == '__main__':
    main()
