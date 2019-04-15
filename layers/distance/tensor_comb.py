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

class TensorComb(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(TensorComb, self).__init__(**kwargs)

    def get_config(self):
        config = {'kernel': self.kernel}
        base_config = super(TensorComb, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.

        self.kernel = self.add_weight(name='kernel',
                                      shape=(int(input_shape[0][1]), int(input_shape[0][1])),
                                      initializer='uniform',
                                      trainable=True)
        super(TensorComb, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        x,y = inputs
        
        #x*W*y'
        res = K.dot(x, self.kernel)
        output = K.batch_dot(K.expand_dims(res,2), K.expand_dims(res,1), axes = (1,2))
        
#        norm1 = K.sqrt(0.00001+ K.sum(x**2, axis = self.axis, keepdims = False))
#        norm2 = K.sqrt(0.00001+ K.sum(y**2, axis = self.axis, keepdims = False))
#        output= K.sum(self.dropout_probs(x*y),1) / norm1 /norm2
    
         

        return K.squeeze(output, axis = 2)

    def compute_output_shape(self, input_shape):
#        print(input_shape)
        # print(type(input_shape[1]))
        output_shape = [input_shape[0],1]

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
    

    x =  Input(shape=(10,))
    y =  Input(shape=(10,))

    output = TensorComb()([x,y])

    encoder = Model([x,y], output)
    encoder.compile(loss = 'mean_squared_error',
            optimizer = 'rmsprop',
            metrics=['accuracy'])
#    
    a = np.random.random((20,10))
    b = np.random.random((20,10))
    c = np.random.random((20,1))

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




