# -*- coding: utf-8 -*-

import numpy as np
import six
import keras
from keras import backend as K
from keras.losses import *
from keras.layers import Lambda
from keras.utils.generic_utils import deserialize_keras_object
import tensorflow as tf

#mz_specialized_losses = {'rank_hinge_loss', 'rank_crossentropy_loss'}


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


def rank_crossentropy_loss(kwargs=None):
    neg_num = 1
    if isinstance(kwargs, dict) and 'neg_num' in kwargs:
        neg_num = kwargs['neg_num']

    def _cross_entropy_loss(y_true, y_pred):
        y_pos_logits = Lambda(lambda a: a[::(neg_num+1), :], output_shape= (1,))(y_pred)
        y_pos_labels = Lambda(lambda a: a[::(neg_num+1), :], output_shape= (1,))(y_true)
        logits_list, labels_list = [y_pos_logits], [y_pos_labels]
        for i in range(neg_num):
            y_neg_logits = Lambda(lambda a: a[(i+1)::(neg_num+1), :], output_shape= (1,))(y_pred)
            y_neg_labels = Lambda(lambda a: a[(i+1)::(neg_num+1), :], output_shape= (1,))(y_true)
            logits_list.append(y_neg_logits)
            labels_list.append(y_neg_labels)
        logits = tf.concat(logits_list, axis=1)
        labels = tf.concat(labels_list, axis=1)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return _cross_entropy_loss






def identity_loss(y_true, y_pred):

    return K.mean(y_pred)


def pointwise_loss(y_true, y_pred):
    
    return K.mean(y_pred)


def hinge(y_true, y_pred):
    y_pred= y_pred*2-1
    return K.mean(K.maximum(0.5 - y_true * y_pred, 0.), axis=-1)


def batch_pairwise_oss(y_true, y_pred):
    pos = K.mean(y_true * y_pred, axis=-1)
    neg = K.mean((1. - y_true) * y_pred, axis=-1)
    return K.maximum(neg - pos + 0.1, 0.)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(neg - pos + 1., 0.)



def percision_bacth(y_true, y_pred):
    return K.mean(K.cast(K.equal(y_pred,0),"float32"))


#def serialize(rank_loss):
#    return rank_loss.__name__