# -*- coding: utf-8 -*-
from params import Params
import models
import dataset
import units
from keras import optimizers
from save import save_experiment
import itertools
import argparse

gpu_count = len(units.get_available_gpus())


def run(params,reader):
    params=dataset.process_embedding(reader,params)
    qdnn = models.setup(params)
    model = qdnn.getModel()
    
    model.compile(loss = params.loss,
    #          optimizer = sgd,
              optimizer = units.getOptimizer(name=params.optimizer,lr=params.lr),
              metrics=['accuracy'])
    
    model.summary()
    # word2id = reader.embedding_params['word2id']
    # file_name = 'sentiment_dic/sentiment_dic.txt'
    # pretrain_x = []
    # pretrain_y = []
    # with codecs.open(file_name, 'r') as f:
    #     for line in f:
    #         word, polarity = line.split()
    #         if word in word2id:
    #             word_id = word2id[word]
    #             pretrain_x.append([word_id]* reader.max_sentence_length)
    #             pretrain_y.append(int(float(polarity)))
    
    # pretrain_x = np.asarray(pretrain_x)
    # pretrain_y = to_categorical(pretrain_y)
    
    (train_x, train_y),(test_x, test_y),(val_x, val_y) = reader.get_processed_data()
    
    #model.fit(x=pretrain_x, y = pretrain_y, batch_size = params.batch_size, epochs= 3,validation_data= (test_x, test_y))
    
    history = model.fit(x=train_x, y = train_y, batch_size = params.batch_size, epochs= params.epochs,validation_data= (test_x, test_y))
    
    evaluation = model.evaluate(x = val_x, y = val_y)
    #save_experiment(model, params, evaluation, history, reader, config_file)

grid_parameters ={
#        "dataset_name":["MR","TREC","SST_2","SST_5","MPQA","MRPC","SUBJ","STS","SNLI","CR"],
        "wordvec_path":["glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt","glove/glove.6B.200d.txt","glove/glove.6B.300d.txt"],
        "loss": ["categorical_crossentropy","categorical_hinge","kullback_leibler_divergence"],
#        "optimizer":["sgd ","rmsprop", "adagrad","adadelta,""adam","adamax","nadam"],
#        "batch_size":[16,32,64,128],
#        "activation":["tanh","relu","sigmoid"],
#        "amplitude_l2":[0.0000005,0.0000001,0.00000005,0.000001,0],
#        "phase_l2":[0.0000005,0.0000001,0.00000005,0.000001,0],
#        "dense_l2":[0.0001,0.00001,0.000001,0],
#        "measurement_size" :[5,10,20,50,100],
#        "lr" : [10,2.5,1,0.25,0.001],
#        "dropout_rate_embedding" : [0.5,0.75,0.8,0.9,1],
#        "dropout_rate_probs" : [0.5,0.75,0.8,0.9,1]     
        }
if __name__=="__main__":

 # import argparse
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the gpu num.',default=0)
    args = parser.parse_args()
    
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
     
            
    params = Params()
    config_file = 'config/qdnn.ini'    # define dataset in the config
    params.parse_config(config_file)
    reader=dataset.setup(params)
    for parameter in parameters:
        params.setup(zip(grid_parameters.keys(),parameter))
#        params.print()
        run(params,reader)
    