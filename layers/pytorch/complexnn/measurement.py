# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn


class ComplexMeasurement(torch.nn.Module):
    def __init__(self, embed_dim, units=5):
        super(ComplexMeasurement, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        kernel = torch.nn.Parameter(torch.rand(self.units, embed_dim, 2))
        self.kernel = F.normalize(kernel.view(self.units, -1), p=2, dim=1, eps=1e-10).view(self.units, embed_dim, 2)

    def forward(self, inputs, measure_operator = None):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        
        batch_size = inputs[0].size(0)
        if measure_operator is None:
            kernel_real = self.kernel[:,:,0].unsqueeze(-1)
            kernel_imag = self.kernel[:,:,1].unsqueeze(-1)
        else:
            kernel_real = measure_operator[0].unsqueeze(-1)
            kernel_imag = measure_operator[1].unsqueeze(-1)
        
        input_real = inputs[0]
        input_imag = inputs[1]
        
        projector_real = torch.bmm(kernel_real, kernel_real.transpose(1, 2)) \
            + torch.bmm(kernel_imag, kernel_imag.transpose(1, 2))
            
        projector_imag = torch.bmm(kernel_imag, kernel_real.transpose(1, 2)) \
            - torch.bmm(kernel_real, kernel_imag.transpose(1, 2))
        
        output_real = torch.mm(input_real.view(batch_size, self.embed_dim*self.embed_dim), projector_real.view(self.units,self.embed_dim*self.embed_dim).t())\
        - torch.mm(input_imag.view( batch_size, self.embed_dim*self.embed_dim), projector_imag.view(self.units,self.embed_dim*self.embed_dim).t())
        
#        output_imag = torch.mm(input_real.view(batch_size, self.embed_dim*self.embed_dim), projector_imag.view(self.units,self.embed_dim*self.embed_dim).t())\
#        + torch.mm(input_imag.view(batch_size, self.embed_dim*self.embed_dim), projector_real.view(self.units,self.embed_dim*self.embed_dim).t())
          
        return output_real
    
if __name__ == '__main__':
    model = ComplexMeasurement(6, units=3)
    a = torch.randn(5,6,6)
    b = torch.randn(5,6,6)

    y_pred = model([a,b])
    print(y_pred[0])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    