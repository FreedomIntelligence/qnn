# -*- coding: utf-8 -*-

from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()    
#tf.executing_eagerly()
from keras.models import Model, Input
input_1 = Input(shape=(3,5), dtype='float')

b = Activation('softmax')(input_1)

model = Model(input_1, b)
a= np.random.random((1,3,5))

model.predict(a)