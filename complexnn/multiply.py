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

class ComplexMultiply(Layer):
    # Input is [phase_embedding, amplitude_embedding]
    # Output is [sentence_embedding_real, sentence_embedding_imag]
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        self.trainable = False
        super(ComplexMultiply, self).__init__(**kwargs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(ComplexMultiply, self).get_config()
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
        super(ComplexMultiply, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]


        sentence_length = amplitude.shape[1]
        embedding_dim = amplitude.shape[2]

        if(len(phase.shape) == 2):
            cos = K.repeat_elements(K.cos(phase), embedding_dim, axis = 2)
            sin = K.repeat_elements(K.sin(phase), embedding_dim, axis = 2)
        elif(len(phase.shape) == 3):
            cos = K.cos(phase)
            sin = K.sin(phase)
        else:
            raise ValueError('Each input should be of dimension 2 or 3.'
                            'Got ' + str(len(phase.shape)) + ' dimension.')

#        print(cos.shape)
#        print(sin.shape)

        real_part = cos*amplitude
        imag_part = sin*amplitude
        # print(real_part.shape)
        # print(imag_part.shape)

        return [real_part,imag_part]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))

        return [input_shape[1], input_shape[1]]


def main():


    input_2 = Input(shape=(3,5), dtype='float')
    input_1 = Input(shape=(3,1), dtype='float')
    [output_1, output_2] = ComplexMultiply()([input_1, input_2])


    model = Model([input_1, input_2], [output_1, output_2])
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    model.summary()

    x = np.random.random((3,3,1))
    x_2 = np.random.random((3,3,5))


    # print(x)
    # print(x_2)
    output = model.predict([x,x_2])
    print(output[0].shape)

if __name__ == '__main__':
    main()
