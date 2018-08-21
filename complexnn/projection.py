import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
from keras.constraints import unit_norm
import tensorflow as tf
import sys
import os
import keras.backend as K
import math


class Complex1DProjection(Layer):

    def __init__(self, dimension, **kwargs):
        # self.output_dim = output_dim
        super(Complex1DProjection, self).__init__(**kwargs)
        self.dimension = dimension

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=(2,self.dimension,1),
                                      constraint = unit_norm(axis = (0,1)),
                                      initializer='uniform',
                                      trainable=True)
        # Create a trainable weight variable for this layer.

        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
             raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')

        super(Complex1DProjection, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        #Implementation of ||<p|v>||^2

        P_real = self.kernel[0,:,:]
        P_imag = self.kernel[1,:,:]

        v_real = inputs[0]
        v_imag = inputs[1]


        # print(K.sum(K.dot(P_real,K.transpose(v_real)),axis = 0))
        Pv_real = K.dot(v_real, P_real)+K.dot(v_imag, P_imag)
        Pv_imag = -K.dot(v_imag, P_real)+K.dot(v_real, P_imag)

        y = K.square(Pv_real)+K.square(Pv_imag)
        # y = K.sum(K.square(v_real))
        return y

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        output_shape = [None,1]
        return(tuple(output_shape))


def main():

    input_1 = Input(shape=(5,), dtype='float')
    input_2 = Input(shape=(5,), dtype='float')
    output = Complex1DProjection(5)([input_1, input_2])


    model = Model([input_1, input_2], output)
    model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    model.summary()

    complex_array = np.random.random((3,5,2))

    norm_2 = np.linalg.norm(complex_array, axis = (1,2))

    for i in range(complex_array.shape[0]):
        complex_array[i] = complex_array[i]/norm_2[i]
    # complex_array()= complex_array / norm_2
    # x_2 = np.random.random((3,5))

    x = complex_array[:,:,0]
    x_2 = complex_array[:,:,1]
    y = np.array([[1],[1],[0]])
    print(x)
    print(x_2)

    for i in range(1000):
        model.fit([x,x_2],y)
    # print(model.get_weights())
        output = model.predict([x,x_2])
        print(output)
    # print(output)








class ComplexProjection(Layer):

    def __init__(self, dimension, **kwargs):
        # self.output_dim = output_dim
        super(ComplexProjection, self).__init__(**kwargs)
        self.dimension = dimension

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.dimension,self.dimension,2),
                                      constraint = None,
                                      initializer='uniform',
                                      trainable=True)
        # Create a trainable weight variable for this layer.

        # if len(input_shape) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on a only one input. '
        #                      'Got ' + str(len(input_shape)) + ' inputs.')


        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(ComplexProjection, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        # if len(inputs) != 1:
        #     raise ValueError('This layer should be called '
        #                      'on only 1 input.'
        #                      'Got ' + str(len(input)) + ' inputs.')

        #Implementation of Tr(P|v><v|) = ||P|v>||^2
        P_real = self.kernel[:,:,0]
        P_imag = self.kernel[:,:,1]

        v_real = inputs[:,:,0]
        v_imag = inputs[:,:,1]

        # print(P_real.shape)
        # print(v_real.shape)
        # print(K.sum(K.dot(P_real,K.transpose(v_real)),axis = 0))
        Pv_real = K.dot(P_real,K.transpose(v_real))-K.dot(P_imag,K.transpose(v_imag))
        Pv_imag = K.dot(P_real,K.transpose(v_imag))+K.dot(P_imag,K.transpose(v_real))
        y = K.sum(K.square(Pv_real), axis = 0)+K.sum(K.square(Pv_imag), axis = 0)
        # print(y)
        return y

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        output_shape = [None,2]
        return([tuple(output_shape)])




if __name__ == '__main__':
    main()
