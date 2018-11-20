# -*- coding: utf-8 -*-

import torch
from torch.nn import Parameter, init
import torch.nn.functional as F
import math

class ComplexDense(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.real_weight = Parameter(torch.Tensor(out_features, in_features))
        self.imag_weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(2 * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.real_weight)
            bound = 1 / math.sqrt(2 * fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        real_input = inputs[0]
        imag_input = inputs[1]

        inputs = torch.cat([real_input, imag_input], dim=1)

        cat_weights_4_real = torch.cat(
            [self.real_weight.t(), -self.imag_weight.t()],
            dim=-1
        )

        cat_weights_4_imag = torch.cat(
            [self.imag_weight.t(), self.real_weight.t()],
            dim=-1
        )
        cat_weights_4_complex = torch.cat(
            [cat_weights_4_real, cat_weights_4_imag],
            dim=0
        )

        output = torch.matmal(inputs, cat_kernels_4_complex)
        # print(output.shape)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )