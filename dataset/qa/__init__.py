# -*- coding: utf-8 -*-
from .data_reader import CRDataReader,MRDataReader,SUBJDataReader,MPQADataReader,SSTDataReader,TRECDataReader
from .data import get_lookup_table
import os
import codecs
import numpy as np
from keras.utils import to_categorical
def setup(opt):
    dir_path = os.path.join(opt.datasets_dir, opt.dataset_name)
    if(opt.dataset_name == 'wiki'):
        reader = CRDataReader(dir_path)
    if(opt.dataset_name == 'trec'):
        reader = MRDataReader(dir_path)
    if(opt.dataset_name == 'nlp'):
        reader = SUBJDataReader(dir_path)
    if(opt.dataset_name == 'MPQA'):
        reader = MPQADataReader(dir_path)
    if(opt.dataset_name == 'SST_2'):
        dir_path = os.path.join(opt.datasets_dir, 'SST')
        reader = SSTDataReader(dir_path, nclasses = 2)
    if(opt.dataset_name == 'SST_5'):
        dir_path = os.path.join(opt.datasets_dir, 'SST')
        reader = SSTDataReader(dir_path, nclasses = 5)
    if(opt.dataset_name == 'TREC'):
        reader = TRECDataReader(dir_path)
       
   
    return reader



def process_embedding(reader,opt):
    opt.max_sequence_length = reader.get_max_sentence_length()

    
    if  opt.wordvec_path == 'random':
        opt.random_init = True
    else:
        opt.random_init = False
        orthonormalized = (opt.wordvec_initialization == "orthonormalized")
        embedding_params = reader.get_word_embedding(opt.wordvec_path,orthonormalized=orthonormalized)
        opt.lookup_table = get_lookup_table(embedding_params)
        opt.idfs = embedding_params["id2idf"]

        
    # print(embedding_params['word2id'])
        
    opt.nb_classes = reader.nb_classes
    return opt

    
    # -*- coding: utf-8 -*-

