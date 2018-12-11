# -*- coding: utf-8 -*-

from layers.keras.complexnn.embedding import phase_embedding_layer, amplitude_embedding_layer
from layers.keras.complexnn.multiply import ComplexMultiply
from layers.keras.complexnn.superposition import ComplexSuperposition
from layers.keras.complexnn.dense import ComplexDense
from layers.keras.complexnn.mixture import ComplexMixture
from layers.keras.complexnn.measurement import ComplexMeasurement
from layers.keras.complexnn.product import ComplexProduct
from layers.keras.complexnn.concatenation import Concatenation
from layers.keras.complexnn.index import Index
from layers.keras.complexnn.ngram import NGram
from layers.keras.complexnn.utils import GetReal
from layers.keras.complexnn.projection import Complex1DProjection
from layers.keras.complexnn.l2_norm import L2Norm
from layers.keras.complexnn.l2_normalization import L2Normalization
from layers.keras.complexnn.utils import *
from layers.keras.complexnn.reshape import reshape
from layers.keras.complexnn.lambda_functions import *
from layers.keras.complexnn.cosine import Cosinse
from layers.keras.complexnn.marginLoss import MarginLoss
from layers.keras.complexnn.AESD import AESD
from layers.keras.complexnn.remove_mask import RemoveMask
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
        raise ImportError('Class {} cannot be found ({})'.format(class_str,traceback.format_exception(*sys.exc_info())))


def getScore(import_str = "", *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)