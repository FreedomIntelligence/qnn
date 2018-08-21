# -*- coding: utf-8 -*-


import os
import numpy as np
from .QAHelper import dataHelper

def setup(opt):
    dir_path = os.path.join(os.path.join(opt.datasets_dir, "QA"),opt.dataset_name)
    
    reader = dataHelper(opt)

       
   
    return reader



def process_embedding(reader,opt):
    
    q_max_sent_length = max(map(lambda x:len(x),reader["train"]['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),reader["train"]['answer'].str.split()))
    
    opt.max_sequence_length = max(q_max_sent_length,a_max_sent_length)
    
#    if  opt.wordvec_path == 'random':
#        opt.random_init = True
#    else:
#        opt.random_init = False
#        orthonormalized = (opt.wordvec_initialization == "orthonormalized")
#        embedding_params = reader.get_word_embedding(opt.wordvec_path,orthonormalized=orthonormalized)
#        opt.lookup_table = get_lookup_table(embedding_params)
#        opt.idfs = embedding_params["id2idf"]

        
    # print(embedding_params['word2id'])
    alphabet = get_alphabet([reader["train"],reader["test"],reader["dev"]])
    print('the number of words',len(alphabet))

    print('get embedding')
    if opt.dataset_name=="NLPCC":     # can be updated
        embeddings = get_embedding(alphabet,language="cn")
    else:
        embeddings = get_embedding(alphabet,fname=opt.wordvec_path)
    opt.embeddings = embeddings
    opt.nb_classes = 2               # for nli, this could be 3
    opt.alphabet=alphabet
    opt.embedding_size = embeddings.shape[1]
     
    return opt

    
    # -*- coding: utf-8 -*-



