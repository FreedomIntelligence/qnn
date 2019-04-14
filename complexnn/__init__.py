# -*- coding: utf-8 -*-

from complexnn.embedding import phase_embedding_layer, amplitude_embedding_layer
from complexnn.multiply import ComplexMultiply
from complexnn.superposition import ComplexSuperposition
from complexnn.dense import ComplexDense
from complexnn.mixture import ComplexMixture
from complexnn.measurement import ComplexMeasurement
from complexnn.concatenation import Concatenation
from complexnn.index import Index
from complexnn.ngram import NGram
from complexnn.utils import GetReal
from complexnn.projection import Complex1DProjection
from complexnn.l2_norm import L2Norm
from complexnn.l2_normalization import L2Normalization
from complexnn.utils import *
from complexnn.reshape import reshape
from complexnn.lambda_functions import *
from complexnn.cosine import Cosine
from complexnn.marginLoss import MarginLoss
from complexnn.AESD import AESD
#def get
import os,sys,traceback
def import_class(import_str):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.insert(0,dirname)
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))


def getScore(import_str = "", *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)