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

class reshape(Layer):

    def __init__(self, shape,**kwargs):
        # self.output_dim = output_dim
        super(reshape, self).__init__(**kwargs)
        self.shape = shape


    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(reshape, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):



        output = K.reshape(inputs,self.shape)
        return output

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))

        return self.shape


def main():
    from keras.layers import Input, Dense


    
    encoding_dim = 50
    input_dim = 300        
    a=np.random.random([10,300])
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim)(input_img) #, 
    new_code=reshape((-1,500))(encoded)
    
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
