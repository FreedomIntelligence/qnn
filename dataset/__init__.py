# -*- coding: utf-8 -*-
from .data_reader import CRDataReader,MRDataReader,SUBJDataReader,MPQADataReader,SSTDataReader,TRECDataReader
from .data import get_lookup_table
import os
def setup(opt):
    dir_path = os.path.join(opt.datasets_dir, opt.dataset_name)
    if(opt.dataset_name == 'CR'):
        reader = CRDataReader(dir_path)
    if(opt.dataset_name == 'MR'):
        reader = MRDataReader(dir_path)
    if(opt.dataset_name == 'SUBJ'):
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
        
    opt.max_sequence_length = reader.get_max_sentence_length()
    if(opt.wordvec_initialization == 'orthogonalize'):
        embedding_params = reader.get_word_embedding(opt.wordvec_path,orthonormalized=True)

    elif( (opt.wordvec_initialization == 'random') | (opt.wordvec_initialization == 'word2vec')):
        embedding_params = reader.get_word_embedding(opt.wordvec_path,orthonormalized=False)
    else:
        raise ValueError('The input word initialization approach is invalid!')
        
    # print(embedding_params['word2id'])
    opt.lookup_table = get_lookup_table(embedding_params)
    opt.random_init = True
    if not(opt.wordvec_initialization == 'random'):
        opt.random_init = False
    opt.nb_classes = reader.nb_classes

    
    return reader
    