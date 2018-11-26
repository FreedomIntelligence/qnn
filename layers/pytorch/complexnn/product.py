# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class ComplexProduct(torch.nn.Module):
    def __init__(self):
        super(ComplexProduct, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        
        left = inputs[0]
        right = inputs[1]
        
        left_real = left[0]
        left_imag = left[1]
        
        right_real = right[0]
        right_imag = right[1]
        
        real_part = left_real * right_real - left_imag * right_imag
        imag_part = left_real * right_imag + left_imag * right_real
        
        return [real_part,imag_part]

def test():
    product = ComplexProduct()
    a = torch.randn(4, 10)
    b = torch.randn(4, 10)
    c = torch.randn(4, 10)
    d = torch.randn(4, 10)
    product = product([[a,b],[c,d]])
    if product[0].size(1) == a.size(1):
        print('ComplexProduct Test Passed.')
    else:
        print('ComplexProduct Test Failed.')

if __name__ == '__main__':
    test()