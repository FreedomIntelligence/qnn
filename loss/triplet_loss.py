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

        positive = y_pred[0]
        negative = y_pred[1]       
     
        basic_loss = positive-negative+margin

        return  K.mean(K.maximum(basic_loss, 0.0),keepdims = True)

    return _margin_loss
def percision(sb,inputs):

    positive = inputs[0]
    negative = inputs[1]


    basic_loss =  (positive-negative)>0
    loss = K.mean(basic_loss)

    return loss


def positive(sb,inputs):

    positive = inputs[0]

    loss = K.mean(positive)

    return loss

def negative(sb,inputs):


    negative = inputs[1]
    loss = K.mean(negative)

    return loss