# -*- coding: utf-8 -*-

import torch
from layers.pytorch.complexnn import *

class Product():
    def __init__(self, opt):
        self.opt = opt
        self.product = ComplexProduct()
    
    def get_representation(self, rep_left, rep_right):
        if type(rep_left) is list: 
            # complex product
            [output_real, output_imag] = self.product([rep_left, rep_right])
            output = [output_real, output_imag]
        else:
            # real product
            # element-wise multiplication
            output = rep_left * rep_right

        return output