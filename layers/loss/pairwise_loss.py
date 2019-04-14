# -*- coding: utf-8 -*-
import keras.backend as K

def identity_loss(y_true, y_pred):

    return K.mean(y_pred)


def pointwise_loss(y_true, y_pred):
    
    return K.mean(y_pred)


def hinge(y_true, y_pred):
    y_pred= y_pred*2-1
    return K.mean(K.maximum(0.5 - y_true * y_pred, 0.), axis=-1)


def batch_pairwise_loss(y_true, y_pred):
    pos = K.mean(y_true * y_pred, axis=-1)
    neg = K.mean((1. - y_true) * y_pred, axis=-1)
    return K.maximum(neg - pos + 0.1, 0.)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(neg - pos + 1., 0.)


