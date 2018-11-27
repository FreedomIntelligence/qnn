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
        
        input_real = inputs[0]
        input_imag = inputs[1] 
        
        #input_real.shape = (batch_size,embedding_dimension,embedding_dimension)
        #input_imag.shape = (batch_size,embedding_dimension,embedding_dimension)
        
        projector_real = torch.bmm(kernel_real.view(self.units, embedding_dimension,1), kernel_real.view(self.units, 1,embedding_dimension)) \
            + torch.bmm(kernel_imag.view(self.units, embedding_dimension,1), kernel_imag.view(self.units, 1,embedding_dimension))
            
        projector_imag = torch.bmm(kernel_imag.view(self.units, embedding_dimension,1), kernel_real.view(self.units, 1,embedding_dimension)) \
            -torch.bmm(kernel_real.view(self.units, embedding_dimension,1), kernel_imag.view(self.units, 1,embedding_dimension))
        
        projector_real = projector_real
        projector_imag = projector_imag
        #projector_real.shape = (self.units,embedding_dimension,embedding_dimension)
        #projector_imag.shape = (self.units,embedding_dimension,embedding_dimension)
        output_real = torch.mm(input_real.view(batch_size, embedding_dimension*embedding_dimension), projector_real.view(self.units,embedding_dimension*embedding_dimension).permute(1,0))\
        - torch.mm(input_imag.view( batch_size, embedding_dimension*embedding_dimension), projector_imag.view(self.units,embedding_dimension*embedding_dimension).permute(1,0))
        
        output_imag = torch.mm(input_real.view(batch_size, embedding_dimension*embedding_dimension), projector_imag.view(self.units,embedding_dimension*embedding_dimension).permute(1,0))\
        + torch.mm(input_imag.view(batch_size, embedding_dimension*embedding_dimension), projector_real.view(self.units,embedding_dimension*embedding_dimension).permute(1,0))
        
       
        output = output_real+output_imag
        return output_real, output_imag
    
if __name__ == '__main__':

        
    from qutip import rand_dm  
    reals, images = [] ,[]
    model = ComplexMeasurement(3)
    for i in range(4):
        m = rand_dm(5)
        images.append( m.data.toarray().imag)
        reals.append( m.data.toarray().real)
    
    x= torch.from_numpy(np.array(reals,dtype="float32"))
    y = torch.from_numpy(np.array(images,dtype="float32"))

    y_pred = model([x,y])
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    