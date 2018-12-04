# -*- coding: utf-8 -*-
from keras.layers import Lambda
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



def precision_batch(y_true, y_pred):
    return K.mean(K.cast(K.equal(y_pred,0),"float32"))

def point_wise_accuracy(y_true, y_pred):
    predicted_label = K.cast(K.greater( y_pred, 0.5),"float32")
    return K.mean(K.cast(K.equal(predicted_label,y_true),"float32"))

    
    