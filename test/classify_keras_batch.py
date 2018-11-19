# -*- coding: utf-8 -*-
from params import Params
from models import representation as models
import dataset
import units
from tools.save import save_experiment
import itertools
import argparse
import keras.backend as K

#gpu_count = len(units.get_available_gpus())
#dir_path,global_logger = units.getLogger()
from tools.logger import Logger
logger = Logger()

def run(params,reader):
    params=dataset.process_embedding(reader,params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    
    model.compile(loss = params.loss,
              optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
              metrics=['accuracy'])
    
    model.summary()    
    (train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()
    
    #pretrain_x, pretrain_y = dataset.get_sentiment_dic_training_data(reader,params)
    #model.fit(x=pretrain_x, y = pretrain_y, batch_size = params.batch_size, epochs= 3,validation_data= (test_x, test_y))
    
    history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,
                        validation_data= (test_x, test_y),callbacks=[logger.getCSVLogger()])
    
    evaluation = model.evaluate(x = val_x, y = val_y)
    save_experiment(model, params, evaluation, history, reader)
    #save_experiment(model, params, evaluation, history, reader, config_file)

    return history,evaluation


grid_parameters ={
        "dataset_name":["MR","TREC","SST_2","SST_5","MPQA","SUBJ","CR"],
        "wordvec_path":["glove/glove.6B.50d.txt"],#"glove/glove.6B.300d.txt"],"glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt",
        "loss": ["categorical_crossentropy"],#"mean_squared_error"],,"categorical_hinge"
        "optimizer":["rmsprop"], #"adagrad","adamax","nadam"],,"adadelta","adam"
        "batch_size":[16],#,32
        "activation":["sigmoid"],
        "amplitude_l2":[0], #0.0000005,0.0000001,
        "phase_l2":[0.00000005],
        "dense_l2":[0],#0.0001,0.00001,0],
        "measurement_size" :[1400,1600,1800,2000],#,50100],
        "lr" : [0.1],#,1,0.01
        "dropout_rate_embedding" : [0.9],#0.5,0.75,0.8,0.9,1],
        "dropout_rate_probs" : [0.9]#,0.5,0.75,0.8,1]     
    }

grid_parameters ={
        "dataset_name":["SST_2"],
        "wordvec_path":["glove/glove.6B.50d.txt"],#"glove/glove.6B.300d.txt"],"glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt",
        "loss": ["categorical_crossentropy"],#"mean_squared_error"],,"categorical_hinge"
        "optimizer":["rmsprop"], #"adagrad","adamax","nadam"],,"adadelta","adam"
        "batch_size":[16],#,32
        "activation":["sigmoid"],
        "amplitude_l2":[0], #0.0000005,0.0000001,
        "phase_l2":[0.00000005],
        "dense_l2":[0],#0.0001,0.00001,0],
        "measurement_size" :[30],#,50100],
        "lr" : [0.1],#,1,0.01
        "dropout_rate_embedding" : [0.9],#0.5,0.75,0.8,0.9,1],
        "dropout_rate_probs" : [0.9],#,0.5,0.75,0.8,1]    ,
        "ablation" : [1],
#        "network_type" : ["ablation"]
    }
if __name__=="__main__":

 # import argparse
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
     
    parameters= parameters[::-1]        
    params = Params()
    config_file = 'config/yahoo.ini'    # define dataset in the config
    params.parse_config(config_file)    
    for parameter in parameters:
        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(),parameter))
        if old_dataset != params.dataset_name:
            print("switch %s to %s"%(old_dataset,params.dataset_name))
            reader=dataset.setup(params)
            params.reader = reader
#        params.print()
#        dir_path,logger = units.getLogger()
#        params.save(dir_path)
        history,evaluation=run(params,reader)
        logger.info("%s : %.4f "%( params.to_string() ,max(history.history["val_acc"])))
        K.clear_session()


