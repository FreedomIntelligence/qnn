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

class L2Norm(Layer):

    def __init__(self, axis = 1, keep_dims = True, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        self.keep_dims = keep_dims
        super(L2Norm, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis, 'keep_dims': self.keep_dims}
        base_config = super(L2Norm, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(L2Norm, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):



        output = K.sqrt(0.00001+K.sum(inputs**2, axis = self.axis, keepdims = self.keep_dims))

        return output

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))

        return input_shape


def main():
    from keras.layers import Input, Dense



    encoding_dim = 50
    input_dim = 300
    a=np.random.random([5,300])
    input_img = Input(shape=(input_dim,))
    # encoded = Dense(encoding_dim)(input_img) #,
    new_code=L2Norm()(input_img)

    encoder = Model(input_img, new_code)

    b=encoder.predict(a)
    print(b)
    print(np.linalg.norm(a,axis=1))



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
