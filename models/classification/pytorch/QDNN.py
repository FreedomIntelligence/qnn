# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.pytorch.complexnn import *

class QDNN(nn.Module):
#    def __init__(self, embedding_matrix, num_measurements):
    def __init__(self, opt):
        """
        max_sequence_len: input sentence length
        embedding_dim: input dimension
        num_measurements: number of measurement units, also the output dimension

        """
        super(QDNN, self).__init__()
        self.max_sequence_len = opt.max_sequence_length
        self.num_measurements = opt.measurement_size
#        self.max_sequence_len = embedding_matrix.shape[1]
        self.embedding_matrix = nn.Parameter(torch.tensor(opt.lookup_table).permute(1,0)).detach()
#        self.embedding_matrix = np.transpose()
        self.embedding_dim = self.embedding_matrix.shape[0]
        self.phase_embedding_layer = PhaseEmbedding(self.max_sequence_len, self.embedding_dim)
        print(self.embedding_matrix.shape)
        self.amplitude_embedding_layer = AmplitudeEmbedding(self.embedding_matrix, random_init = False)
        self.l2_norm = L2Norm(dim = -1, keep_dims = False)
        self.l2_normalization = L2Normalization(dim = -1)
        self.activation = nn.Softmax(dim = -1)
        self.complex_multiply = ComplexMultiply()
        self.mixture = ComplexMixture(use_weights = True)
        self.measurement = ComplexMeasurement(units = self.num_measurements)
#            self.output = ComplexMeasurement(units = self.opt.measurement_size)([self.sentence_embedding_real, self.sentence_embedding_imag])
    def forward(self, input_seq):
        """
        In the forward function we accept a Variable of input data and we must 
        return a Variable of output data. We can use Modules defined in the 
        constructor as well as arbitrary operators on Variables.
        """
        phase_embedding = self.phase_embedding_layer(input_seq)
        amplitude_embedding = self.amplitude_embedding_layer(input_seq)
        weights = self.l2_norm(amplitude_embedding)
        amplitude_embedding = self.l2_normalization(amplitude_embedding)
        weights = self.activation(weights)
        [seq_embedding_real, seq_embedding_imag] = self.complex_multiply([phase_embedding, amplitude_embedding])
        [sentence_embedding_real, sentence_embedding_imag] = self.mixture([seq_embedding_real, seq_embedding_imag,weights])
        output = self.measurement([sentence_embedding_real, sentence_embedding_imag])
        
        return output