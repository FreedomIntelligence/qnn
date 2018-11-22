# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

class ComplexMultiply(torch.nn.Module):
    # Input is [phase_embedding, amplitude_embedding]
    # Output is [sentence_embedding_real, sentence_embedding_imag]
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]


        embedding_dim = amplitude.size(-1)
        
        if len(amplitude.size()) == len(phase.size())+1: # Assigning each dimension with same phase
            cos = torch.unsqueeze(torch.cos(phase), dim=-1)
            sin = torch.unsqueeze(torch.sin(phase), dim=-1)
            
        elif len(amplitude.shape) == len(phase.shape): #Each dimension has different phases
            cos = torch.cos(phase)
            sin = torch.sin(phase)
        
       
        else:
             raise ValueError('input dimensions of phase and amplitude do not agree to each other.')

        real_part = cos*amplitude
        imag_part = sin*amplitude

        return [real_part, imag_part]