# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

from keras.layers import Embedding, GlobalAveragePooling1D,Dense, Masking, Flatten,Dropout, Activation
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()    
#tf.executing_eagerly()
from keras.models import Model, Input
import numpy as np
import keras.backend as K
input_1 = Input(shape=(3,5), dtype='float')

b = Activation('softmax')(input_1)
embedding = tf.Variable(  tf.random_uniform([128,100], -1.0, 1.0))

model = Model(input_1, b)
a= np.random.random((1,3,5))

model.predict(a)
import codecs
with codecs.open("words1.txt","w",encoding="utf-8")  as f:
    for k,v in count.items():
        if v<5:
            f.write("%s : %d"%(k,v))
from complexnn.l2_normalization import L2normalization
encoding_dim = 50
input_dim = 300        
a=np.random.random([5,300])
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim)(input_img) #, 
new_code=L2normalization(encoded)

encoder = Model(input_img, new_code)

b=encoder.predict(a)
np.linalg.norm(b,axis=1)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalization_op = tf.assign(embeddings,  embeddings / norm) 