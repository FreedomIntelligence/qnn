# -*- coding: utf-8 -*-
from keras.layers import Lambda
import keras.backend as K

def rank_hinge_loss(kwargs=None):
    margin = 1.
    if isinstance(kwargs, dict) and 'margin' in kwargs:
        margin = kwargs['margin']
    def _margin_loss(y_true, y_pred):
# output_shape = K.int_shape(y_pred)
        y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
        y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
        loss = K.maximum(0., margin + y_neg - y_pos)
        return K.mean(loss)

    return _margin_loss
