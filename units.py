# -*- coding: utf-8 -*-
import keras
import numpy as np
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

def getLogger():
    import random,logging,time,sys,os
    random_str = str(random.randint(1,10000))
    
    now = int(time.time()) 
    timeArray = time.localtime(now)
    timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
    log_path = "log/acc" +time.strftime("%Y%m%d", timeArray)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program) 
    
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',datefmt='%a, %d %b %Y %H:%M:%S',filename=log_path+'/'+timeStamp+"_"+ random_str+'.log',filemode='w')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    return log_path,logger

def batch_softmax_with_first_item(x):
    x_exp = np.exp(x)
    x_sum = np.repeat(np.expand_dims(np.sum(x_exp, axis=1),1), x.shape[1], axis=1)
    return x_exp / x_sum

def to_array(ll,maxlen=0,use_mask=False):
    lens = [len(l) for l in ll]
    if maxlen ==0:
        maxlen=max(lens)
    arr = np.zeros((len(ll),maxlen),int)
    mask = np.arange(maxlen) < np.array(lens)[:,None]
    if type(ll[0]) == tuple:
        input_num = len(ll[0])
        out = []
        s = [[i[dd] for i in ll] for dd in range(input_num)]
        for l in s:
            out.append(np.asarray(np.ma.row_stack(l)))
        return out
#    print(mask)
#    print(mask.shape)
    arr[mask] = np.concatenate(ll)
    if use_mask:
        return arr,mask
    return arr