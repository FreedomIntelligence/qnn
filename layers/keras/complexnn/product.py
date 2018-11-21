# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf
import sys
import os
import keras.backend as K
import math

class ComplexProduct(Layer):
    # Input is [phase_embedding, amplitude_embedding]
    # Output is [sentence_embedding_real, sentence_embedding_imag]
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        self.trainable = False
        super(ComplexProduct, self).__init__(**kwargs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(ComplexProduct, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
             raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')


        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(ComplexProduct, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        
        left = inputs[0]
        right = inputs[1]
        
        left_real = left[0]
        left_imag = left[1]
        
        right_real = right[0]
        right_imag = right[1]  
        
        real_part = K.dot(left_real*right_real)-K.dot(left_real*right_real)
        imag_part = K.dot(left_real*right_imag)+K.dot(left_imag*right_real)
        
        return [real_part,imag_part]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        
#        print('Input shape of multiply layer:{}'.format(input_shape))
#        print([input_shape[1], input_shape[1]])
        return [input_shape[1], input_shape[1]]