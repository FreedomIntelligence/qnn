# -*- coding: utf-8 -*-

from layers.pytorch.complexnn.embedding import PhaseEmbedding, AmplitudeEmbedding, ComplexEmbedding
from layers.pytorch.complexnn.multiply import ComplexMultiply
from layers.pytorch.complexnn.superposition import ComplexSuperposition
from layers.pytorch.complexnn.dense import ComplexDense
from layers.pytorch.complexnn.mixture import ComplexMixture
from layers.pytorch.complexnn.product import ComplexProduct
from layers.pytorch.complexnn.measurement import ComplexMeasurement
from layers.pytorch.complexnn.concatenation import Concatenation
from layers.pytorch.complexnn.index import Index
from layers.pytorch.complexnn.ngram import NGram
#from layers.pytorch.complexnn.projection import Complex1DProjection
from layers.pytorch.complexnn.l2_norm import L2Norm
from layers.pytorch.complexnn.l2_normalization import L2Normalization
#from layers.pytorch.complexnn.utils import *
#from layers.pytorch.complexnn.reshape import reshape
from layers.pytorch.complexnn.lambda_functions import *
from layers.pytorch.complexnn.cosine import Cosine
from layers.pytorch.complexnn.margin_loss import MarginLoss
