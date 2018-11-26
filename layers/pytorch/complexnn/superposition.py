# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class ComplexSuperposition(torch.nn.Module):

    def __init__(self, average_weights = False):
        super(ComplexSuperposition, self).__init__()
        self. average_weights = average_weights

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2/3 inputs.')

        if len(inputs) != 3 and len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2/3 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        input_real = inputs[0]
        input_imag = inputs[1]
        
        ndims = 3
        if self.average_weights:
            output_r = torch.mean(input_real, dim=1)
            output_i = torch.mean(input_imag, dim=1)
        else:
            if inputs[2].dim() == 2:
                weight = torch.unsqueeze(inputs[2], dim=-1)
            else:
                weight = inputs[2]

            output_real = input_real * weight #shape: (None, 60, 300)
            output_real = torch.sum(output_real, dim=1) #shape: (None, 300)
            output_imag = input_imag * weight
            output_imag = torch.sum(output_imag, dim=1)
        
        
        output_real_transpose = torch.unsqueeze(output_real, dim=1) #shape: (None, 1, 300)
        output_imag_transpose = torch.unsqueeze(output_imag, dim=1)
        
        output_real = torch.unsqueeze(output_real, dim=2) #shape: (None, 300, 1)
        output_imag = torch.unsqueeze(output_imag, dim=2)

        # output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
        output_r = torch.matmul(output_real, output_real_transpose) + torch.matmul(output_imag, output_imag_transpose) #shape: (None, 300, 300)
        output_i = torch.matmul(output_imag, output_real_transpose) - torch.matmul(output_real, output_imag_transpose) #shape: (None, 300, 300)
        return [output_r, output_i]

def test():
    sup = ComplexSuperposition()
    a = torch.randn(4, 10, 2)
    b = torch.randn(4, 10, 2)
    c = torch.randn(4, 10)
    sup_ = sup([a,b,c])
    print(sup_[0].size())
    if sup_[0].size(1) == a.size(2):
        print('ComplexSuperposition Test Passed.')
    else:
        print('ComplexSuperposition Test Failed.')

if __name__ == '__main__':
    test()