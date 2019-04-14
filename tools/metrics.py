# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 08:42:49 2019

@author: quartz
"""

import keras.backend as K
def precision_batch(y_true, y_pred):
    return K.mean(K.cast(K.equal(y_pred,0),"float32"))
