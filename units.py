# -*- coding: utf-8 -*-
import keras
#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

def getOptimizer(name="sgd",lr=0.0001):
    name=name.strip().lower()
    if name=="sgd":
        optimizer=keras.optimizers.SGD(lr=lr*0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif name=="rmsprop":
        optimizer=keras.optimizers.RMSprop(lr=lr*0.001, rho=0.9, epsilon=None, decay=0.0)
    elif name=="adagrad":
        optimizer=keras.optimizers.Adagrad(lr=lr*0.01, epsilon=None, decay=0.0)
    elif name=="adadelta":
        optimizer=keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
    elif name=="adam":
        optimizer=keras.optimizers.Adam(lr=lr*0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)     
    elif name=="adamax":
        optimizer=keras.keras.optimizers.Adamax(lr=lr*0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)     
    elif name=="nadam":
        optimizer=keras.optimizers.Nadam(lr=lr*0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    else:
        raise Exception("optimizer not supported: {}, only support sgd,rmsprop,adagrad,adadelta,adam,adamax,nadam".format(name))
    return optimizer



def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']