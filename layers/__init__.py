# -*- coding: utf-8 -*-

from layers.cvnn.embedding import phase_embedding_layer, amplitude_embedding_layer
from layers.cvnn.multiply import ComplexMultiply
from layers.cvnn.superposition import ComplexSuperposition
from layers.cvnn.dense import ComplexDense
from layers.cvnn.mixture import ComplexMixture
from layers.cvnn.measurement import ComplexMeasurement
from layers.concatenation import Concatenation
from layers.index import Index
from layers.ngram import NGram
from layers.cvnn.utils import GetReal
from layers.cvnn.projection import Complex1DProjection
from layers.l2_norm import L2Norm
from layers.l2_normalization import L2Normalization
from layers.cvnn.utils import *
from layers.reshape import reshape
from layers.loss.lambda_functions import *
from layers.distance.cosine import Cosinse
from layers.loss.marginLoss import MarginLoss
from layers.distance.AESD import AESD
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


