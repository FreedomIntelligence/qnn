# -*- coding: utf-8 -*-
import sys; sys.path.append('.')
import numpy as np
from keras import backend as K
from keras.layers import Layer,Dense,Dropout
from keras.models import Model, Input
import tensorflow as tf
import sys
import os
import keras.backend as K
import math

class Cosinse(Layer):

    def __init__(self, dropout_keep_prob = 1, axis = -1, keep_dims = True, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        self.keep_dims = keep_dims
        self.dropout_keep_prob=dropout_keep_prob
        self.dropout_probs = Dropout(dropout_keep_prob)
        super(Cosinse, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis, 'keep_dims': self.keep_dims}
        base_config = super(Cosinse, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(Cosinse, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        x,y = inputs

        norm1 = K.sqrt(0.00001+ K.sum(x**2, axis = self.axis, keepdims = False))
        norm2 = K.sqrt(0.00001+ K.sum(y**2, axis = self.axis, keepdims = False))
        output= K.sum(self.dropout_probs(x*y),1) / norm1 /norm2


        return K.expand_dims(output)

    def compute_output_shape(self, input_shape):
#        print(input_shape)
        # print(type(input_shape[1]))
        output_shape = []
        if self.axis<0:
            self.axis = len(input_shape[0])+self.axis 
        for i in range(len(input_shape[0])):            
            if not i == self.axis:
                output_shape.append(input_shape[0][i])
        if self.keep_dims:
            output_shape.append(1)
#        print('Input shape of L2Norm layer:{}'.format(input_shape))
#        print(output_shape)
        return([tuple(output_shape)])


if __name__ == '__main__':
    from keras.layers import Input, Dense

#    encoding_dim = 50

#    input_img = Input(shape=(300,))
#    n = Dense(20)(input_img)
#    print(n.shape)
#    new_code = L2Norm(axis = 1, keep_dims =False)(n)
##    output = Dense(2)(new_code) #,
##    print(output.shape)
#    print(new_code.shape)
#    encoder = Model(input_img, new_code)
#    
#    encoder.compile(loss = 'mean_squared_error',
#            optimizer = 'rmsprop',
#            metrics=['accuracy'])
#    
#    a = np.random.random((5,300))
#    print(encoder.predict(x = a))
#    b = np.random.random((5))
#    encoder.fit(x=a, y=b, epochs = 10)
    

    x =  Input(shape=(2,))
    y =  Input(shape=(2,))

    output = Cosinse()([x,y])

    encoder = Model([x,y], output)
    encoder.compile(loss = 'mean_squared_error',
            optimizer = 'rmsprop',
            metrics=['accuracy'])
#    
    a = np.random.random((5,300))
    b = np.random.random((5,300))
    c = np.random.random((5,1))
    a = np.ones((5,300))
#    b = np.ones((5,300))
#    encoder.fit(x=[a,b], y=c, epochs = 10)
    
    a= np.array([[1,1],[3,4]])
    b= np.array([[1,0],[4,3]])
    print(encoder.predict(x = [a,b]))
    
#    b = np.random.random((5,2,2))
#    encoder.fit(x=a, y = b, epochs = 10)
#    print(b)
#    print(np.linalg.norm(a,axis=1))



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




