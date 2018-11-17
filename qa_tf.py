# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tools.timer import log_time_delta
import datetime

from params import Params
from dataset import qa
from models.match import tensorflow as models
from tools import evaluation
from dataset.qa import QAHelper as helper
from tools.logger import Logger
logger = Logger() 

params = Params()
config_file = 'config/qa.ini'    # define dataset in the config
params.parse_config(config_file)
reader = qa.setup(params)
#params = qa.process_embedding(reader,params)

@log_time_delta
def predict(model,sess,batch,test):
    scores = []
    for data in batch:            
        score = model.predict(sess,data)
        scores.extend(score)  
    return np.array(scores[:len(test)])

best_p1=0
with tf.Graph().as_default(): # ,tf.device("/cpu:" + str(params.gpu))
    # with tf.device("/cpu:0"):
    session_conf = tf.ConfigProto()
    session_conf.allow_soft_placement = True
    session_conf.log_device_placement = False
    session_conf.gpu_options.allow_growth = True
#    sess = tf.Session(config=session_conf)
    sess = tf.InteractiveSession()
    
    model = models.setup(params)
    model.build_graph()   
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    for i in range(params.epochs):  
        
        for data in reader.get_train(overlap_feature=False):#model=model,sess=sess,
#        for data in data_helper.getBatch48008(train,alphabet,args.batch_size):
            _, summary, step, loss, accuracy,score12, score13, see = model.train(sess,data)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))
#            logger.info("{}: step {}, loss {:g}, acc {:g} ,positive {:g},negative {:g}".format(time_str, step, loss, accuracy,np.mean(score12),np.mean(score13)))

        test_datas = reader.get_test()
        predicted_test = predict(model,sess,test_datas,reader.datas["test"])
        map_mrr_test = evaluation.evaluationBypandas(reader.datas["test"],predicted_test)

#        logger.info('map_mrr test' +str(map_mrr_test))
        print('epoch '+ str(i) + ' map_mrr test' +str(map_mrr_test))
        
 