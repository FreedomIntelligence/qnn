# -*- coding: utf-8 -*-
from keras.layers import Lambda
import keras.backend as K

def rank_hinge_loss(kwargs=None):
    margin = 1.
    if isinstance(kwargs, dict) and 'margin' in kwargs:
        margin = kwargs['margin']
    def _margin_loss(y_true, y_pred):
# output_shape = K.int_shape(y_pred)
#        y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
#        y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
#        loss = K.maximum(0., margin + y_neg - y_pos)
        anchor = y_pred[0]
        positive = y_pred[1]
        negative = y_pred[2]
    
        pos_dist = K.sum((anchor-positive)**2, keepdims = True)
        neg_dist = K.sum((anchor-negative)**2, keepdims = True)
        basic_loss = pos_dist-neg_dist+margin
        basic_loss = K.mean(K.maximum(basic_loss, 0.0),keepdims = True)


        return K.mean(basic_loss)

    return _margin_loss
def percision(sb,inputs):
    anchor = inputs[0]
    positive = inputs[1]
    negative = inputs[2]

    pos_dist = K.sum((anchor-positive)**2, keepdims = True)
    neg_dist = K.sum((anchor-negative)**2, keepdims = True)

    basic_loss =  (pos_dist-neg_dist)>0
    loss = K.mean(basic_loss)

    return loss