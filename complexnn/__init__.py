# -*- coding: utf-8 -*-

from .embedding import phase_embedding_layer, amplitude_embedding_layer
from .multiply import ComplexMultiply
from .superposition import ComplexSuperposition
from .dense import ComplexDense
from .mixture import ComplexMixture
from .measurement import ComplexMeasurement
from .concatenation import Concatenation
from .index import Index
from .ngram import NGram
from .utils import GetReal
from .projection import Complex1DProjection
from .l2_norm import L2Norm
from .l2_normalization import L2Normalization
from .utils import *
from .reshape import reshape
from .lambda_functions import *
import numpy as np
from .cosine import Cosinse
from .marginLoss import MarginLoss