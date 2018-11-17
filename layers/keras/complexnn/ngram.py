# -*- coding: utf-8 -*-
import sys; sys.path.append('.')
from keras.layers import Input,Layer
from keras.models import Model
import numpy as np
import math
import keras.backend as K

class NGram(Layer):
    '''
    Input can be a sequence of indexes or a sequence of embeddings
    n_value is the value of n
    axis is the dimension to which n-gram is applied
    
    e.g. input_shape = (None,10) n_value = 5 ==> output_shape = (None,10,5)
    
    e.g. input_shape = (None,10,3) n_value = 5, axis = 1 ==> output_shape = (None,10,5,3)
    
    '''
    def __init__(self, n_value = 3, axis = 1, **kwargs):
        self.n_value = n_value
        self.axis = axis
        super(NGram, self).__init__(**kwargs)

    def get_config(self):
        config = {'n_value': self.n_value, 'axis': self.axis}
        base_config = super(NGram, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        super(NGram, self).build(input_shape)  # Be sure to call this somewhere!
        
    def call(self, inputs):
        
        ndims = len(inputs.shape)
#        print(inputs.shape[1])
        slice_begin_index = [0]*ndims
        slice_end_index = [-1]*ndims
        seq_len = inputs.shape[self.axis]
        list_of_ngrams = []
        
        for i in range(self.n_value):
            begin = max(0,i-math.floor(self.n_value/2))
            end = min(seq_len-1+i-math.floor(self.n_value/2),seq_len-1)
#            print(begin,end)
            slice_begin_index[self.axis] = begin
            slice_end_index[self.axis] = end-begin+1
            l =  K.slice(inputs, slice_begin_index, slice_end_index)
#            print(l.shape)
            
            slice_begin_index[self.axis] = 0
            slice_end_index[self.axis] = int(seq_len-(end-begin+1))
#            print(slice_end_index)
            
            padded_zeros = K.zeros_like(K.slice(inputs, slice_begin_index, slice_end_index))
#            print(padded_zeros.shape)
            if begin == 0:
                #left_padding
                list_of_ngrams.append(K.expand_dims(K.concatenate([padded_zeros,l],axis = self.axis),axis = self.axis+1))
                #right_padding
            else:
                list_of_ngrams.append(K.expand_dims(K.concatenate([l,padded_zeros],axis = self.axis),axis = self.axis+1))
                
        ngram_mat = K.concatenate(list_of_ngrams,axis = self.axis+1)
        
        return(ngram_mat)


    def compute_output_shape(self, input_shape):
        # print(input_shape)
#        ndims = len(input_shape)+1
        output_shape = [i for i in input_shape]
        output_shape.insert(self.axis+1, self.n_value)
#        print(output_shape)
        return([tuple(output_shape)])

def main():

   input_1 = Input(shape=(10,), dtype='float32')
   output = NGram(n_value = 5)(input_1)

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

