# -*- coding: utf-8 -*-

from keras.layers import Input,Layer
from keras.models import Model
import numpy as np
import math
import keras.backend as K

class NGram(Layer):

    def __init__(self, n_value = 3, **kwargs):
        self.n_value = n_value
        super(NGram, self).__init__(**kwargs)

    def get_config(self):
        config = {'n_value': self.n_value}
        base_config = super(NGram, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        super(NGram, self).build(input_shape)  # Be sure to call this somewhere!
        
    def call(self, inputs):
        # print(inputs.shape[1])
        seq_len = inputs.shape[1]
        list_of_ngrams = []
        for i in range(self.n_value):
            begin = max(0,i-math.floor(self.n_value/2))
            end = min(seq_len-1+i-math.floor(self.n_value/2),seq_len-1)
#            print(begin,end)
            l =  K.slice(inputs, [0,begin], [-1,end-begin+1])
#            print(l.shape)
            padded_zeros = K.zeros_like(K.slice(inputs, [0,0], [-1,int(seq_len-(end-begin+1))]))
#            print(padded_zeros.shape)
            if begin == 0:
                #left_padding
                list_of_ngrams.append(K.expand_dims(K.concatenate([padded_zeros,l])))
                #right_padding
            else:
                list_of_ngrams.append(K.expand_dims(K.concatenate([l,padded_zeros])))
                
        
        ngram_mat = K.concatenate(list_of_ngrams,axis = -1)
#        w_list = []
#        seq_len = inputs.shape[1]
#        # print(math.floor(self.n_value/2))
#        for n in range(self.n_value):
#          w = np.zeros(shape = (seq_len,seq_len))
#          for i in range(seq_len):
#            if (i+n-math.floor(self.n_value/2)>= 0) and (i+n-math.floor(self.n_value/2) < seq_len):
        
#              w[i+n-math.floor(self.n_value/2),i] = 1
#          w_list.append(w)

#        kernel = K.zeros(shape =(self.n_value,inputs.shape[1],inputs.shape[1]))
        # print(np.asarray(w_list).shape)
#        K.set_value(kernel, np.asarray(w_list))

#        output = K.dot(inputs,kernel)
#        output = K.permute_dimensions(output, pattern = (0,2,1))
        # output = K.gather(inputs, (:,[0,-3]))
        # print(output.shape)
        # output = inputs[:,self.index,:]
        return(ngram_mat)
        
    def compute_mask(self, inputs, mask=None):
        
        return None

    def compute_output_shape(self, input_shape):
        # print(input_shape)
        output_shape = [input_shape[0],input_shape[1], self.n_value]
        return([tuple(output_shape)])

def main():

   input_1 = Input(shape=(10,), dtype='float32')
   output = NGram(5)(input_1)

   model = Model(input_1, output)
   model.compile(loss='binary_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])
   model.summary()

   x = np.random.randint(20,size = (3,10))
   y = model.predict(x)
   print(x)
   print(y)
   # print(x[:,3,:])



if __name__ == '__main__':
   main()

