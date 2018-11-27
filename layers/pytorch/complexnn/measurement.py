# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
from layers.keras.complexnn.measurement import ComplexMeasurement as ComplexMeasurement2
import numpy as np
from keras.models import Model, Input

class ComplexMeasurement(torch.nn.Module):

    def __init__(self, units = 5, trainable = True):
        super(ComplexMeasurement, self).__init__()
#        self.average_weights = average_weights
        self.units = units
        

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        
       
        embedding_dimension = inputs[0].shape[-1]
        batch_size = inputs[0].shape[0]
        
        
        self.kernel = torch.nn.Parameter(torch.rand(self.units,embedding_dimension,2))
        
        kernel = F.normalize(self.kernel.view(self.units, embedding_dimension*2), p=2, dim=1, eps=1e-10).view(self.units,embedding_dimension,2)
        #self.kernel.shape = (self.units, embedding_dimension,2)
        
        kernel_real = kernel[:,:,0]
        kernel_imag = kernel[:,:,1]
        #kernel_real.shape = (self.units, embedding_dimension)
        #kernel_imag.shape = (self.units, embedding_dimension)
        
        input_real = inputs[0].double()
        input_imag = inputs[1].double() 
        
        #input_real.shape = (batch_size,embedding_dimension,embedding_dimension)
        #input_imag.shape = (batch_size,embedding_dimension,embedding_dimension)
        
        projector_real = torch.bmm(kernel_real.view(self.units, embedding_dimension,1), kernel_real.view(self.units, 1,embedding_dimension)) \
            + torch.bmm(kernel_imag.view(self.units, embedding_dimension,1), kernel_imag.view(self.units, 1,embedding_dimension))
        projector_imag = torch.bmm(kernel_imag.view(self.units, embedding_dimension,1), kernel_real.view(self.units, 1,embedding_dimension)) \
            - torch.bmm(kernel_real.view(self.units, embedding_dimension,1), kernel_imag.view(self.units, 1,embedding_dimension))
        
        projector_real = projector_real.double()
        projector_imag = projector_imag.double()
        #projector_real.shape = (self.units,embedding_dimension,embedding_dimension)
        #projector_imag.shape = (self.units,embedding_dimension,embedding_dimension)
        output_real = torch.mm(input_real.view(batch_size, embedding_dimension*embedding_dimension), projector_real.view(embedding_dimension*embedding_dimension,self.units))
        - torch.mm(input_imag.view( batch_size, embedding_dimension*embedding_dimension), projector_imag.view(embedding_dimension*embedding_dimension,self.units))
        
        output_imag = torch.add(torch.mm(input_real.view(batch_size, embedding_dimension*embedding_dimension), projector_imag.view(embedding_dimension*embedding_dimension,self.units))
        ,torch.mm(input_imag.view( batch_size, embedding_dimension*embedding_dimension), projector_real.view(embedding_dimension*embedding_dimension,self.units)))
        
       
        output = output_real+output_imag
        return output
    
if __name__ == '__main__':

    
    model= ComplexMeasurement(3)
    
    model_2 = ComplexMeasurement2(3)
    x_1 = np.random.random((4,5,5))
    norm_2 = np.linalg.norm(x_1, axis = (1,2))
    for i in range(x_1.shape[0]):
        x_1[i] = x_1[i] / norm_2[i]
    # for i in range(complex_array.shape[0]):
    #     complex_array[i] = complex_array[i]/norm_2[i]
    # # complex_array()= complex_array / norm_2
    x_2 = np.random.random((4,5,5))
    norm_2 = np.linalg.norm(x_2, axis = (1,2))
    for i in range(x_2.shape[0]):
        x_2[i] = x_2[i] / norm_2[i]
    
    
    
    input_1 = Input(shape=(5,5), dtype='float')
    input_2 = Input(shape=(5,5), dtype='float')
    output = ComplexMeasurement2(3)([input_1,input_2])


    model_2 = Model([input_1,input_2], output)
    model_2.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    model_2.summary()
    
    
    output = model_2.predict([x_1,x_2])
    input_1 = torch.from_numpy(x_1)
    input_2 = torch.from_numpy(x_2)
    y_pred = model([input_1,input_2])
#    input_2 = Input(shape=(4,5,5), dtype='float')
#    output = ComplexMeasurement(3)([input_1,input_2])
#
#
#    model = Model([input_1,input_2], output)
#    model.compile(loss='binary_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])
#    model.summary()
#
#    weights = model.get_weights()
#    x_1 = np.random.random((5,4,5,5))
#    x_2 = np.random.random((5,4,5,5))
#    output = model.predict([x_1,x_2])
#    for i in range(5):
#        xy = x_1[i] + 1j * x_2[i]
#        for j in range(3):
#
#            m= weights[0][j,:,0] + 1j *weights[0][j,:,1]
##            print(np.matmul(xy[0] ,np.outer(m,m)))
##            result = np.absolute(np.trace(np.matmul(xy ,np.outer(m,m))))
#            print(np.trace(np.matmul(xy[0] ,np.outer(m,m))))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    