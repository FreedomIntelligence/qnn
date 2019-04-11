# -*- coding: utf-8 -*-
import sys; sys.path.append('.')
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf
import sys
import os
import keras.backend as K
import math

class MarginLoss(Layer):

    def __init__(self, margin = 1, **kwargs):
        # self.output_dim = output_dim
        self.margin = margin
        super(MarginLoss, self).__init__(**kwargs)

    def get_config(self):
        config = {'margin': self.margin}
        base_config = super(MarginLoss, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(MarginLoss, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        score1,score2 = inputs

        output = K.maximum(score2-score1+self.margin,0)
        return output

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        
        return input_shape[0]


def main():
    from keras.layers import Input, Dense



    encoding_dim = 50
    input_dim = 300
    a=np.random.random([5,300])
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim)(input_img) #,
    new_code=MarginLoss()(encoded)

    encoder = Model(input_img, new_code)

    b=encoder.predict(a)
    print(np.linalg.norm(b,axis=1))



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
