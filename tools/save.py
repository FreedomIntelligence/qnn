# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:48:07 2018

@author: quartz
"""
import codecs
import numpy as np
import os
#import shutil
import time
import pandas as pd
def save_experiment(model, params, evaluation, history, reader):
    
    eval_dir = params.eval_dir
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)  

    now = int(time.time()) 
    timeArray = time.localtime(now)
    timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
    
    params_dir = os.path.join(eval_dir,timeStamp)
    if not os.path.exists(params_dir):
        os.mkdir(params_dir)
        
    config_file = os.path.join(params_dir, 'config.ini')
    params.export_to_config(config_file)
#    shutil.copy2(config_file, os.path.abspath(params_dir))
    save_result(evaluation, history, params_dir)
    save_network(model, reader, params_dir)
    
    
    
def save_result(evaluation, history, output_dir):
    eval_result_file = os.path.join(output_dir,'evaluation')
    history_file = os.path.join(output_dir,'history')
    with codecs.open(eval_result_file, 'w') as f:
        f.write('loss = '+ str(evaluation[0])+ ' accuracy = '+ str(evaluation[1]))
    np.save(history_file, history.history)
    pd.DataFrame(history.history).to_csv("history.csv")

def save_network(model, reader, output_dir):
    id2word = reader.embedding_params['id2word']
    word_sentiment_file = os.path.join(output_dir, 'word_sentiment')
    amplitude_embedding_file =  os.path.join(output_dir, 'amplitude_embedding')
    phase_embedding_file = os.path.join(output_dir, 'phase_embedding')
    weights_file = os.path.join(output_dir, 'weights')
    measurements_file = os.path.join(output_dir, 'measurements')
    id2word_file = os.path.join(output_dir, 'id2word')
    
    
    word_embedding = get_word_embedding(model)
    weights = get_weights(model)
    measurements = get_measurements(model)
    export_word_sentiment_dic(id2word, model, word_sentiment_file)
    np.save(amplitude_embedding_file, word_embedding[0])
    np.save(phase_embedding_file, word_embedding[1])
    np.save(weights_file, weights)
    np.save(measurements_file, measurements)
    np.save(id2word_file, id2word)

    
def export_word_sentiment_dic(id2word, model, file_name):
    file = codecs.open(file_name, 'w')
    for i in range(len(id2word)):
        word = id2word[i]
        sentiment = get_word_sentiment(i+1,model)
        file.write(word + ' '+ str(sentiment[0][0]) +'\n')
        
def get_word_sentiment(word_id, model):
    sentence_length = model.layers[0].input_shape[1]
    input_x = np.asarray([[word_id]* sentence_length])
    output = model.predict(input_x)
    return(output)
    
def get_word_embedding(model):
    weights = model.get_weights()
    amplitude_embedding = weights[0]
    phase_embedding = weights[1]
    return(amplitude_embedding, phase_embedding)
    
def get_weights(model):
    weights = model.get_weights()
    weights = weights[2]
    return(weights)
    
def get_measurements(model):
    weights = model.get_weights()
    measurements = weights[3]
    return(measurements)
    