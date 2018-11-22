from __future__ import division
import os
import io
import logging
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from dataset.classification.data import data_gen,set_wordphase,create_dictionary,get_wordvec,get_index_batch
from keras.utils import to_categorical
from collections import Counter
import preprocess
from preprocess.dictionary import Dictionary
from preprocess.bucketiterator import BucketIterator
from preprocess.embedding import Embedding
from units import to_array

class DataReader(object):
    def __init__(self, train, dev, test, opt, nb_classes):
#        self.data = {'train': train, 'dev': dev, 'test': test}
        for key,value in opt.__dict__.items():
            self.__setattr__(key,value)  
        self.preprocessor = preprocess.setup(opt)
        self.datas = {'train': self.preprocess(train), 'dev': self.preprocess(dev), 'test': self.preprocess(test)}
        self.nb_classes = nb_classes
        self.get_max_sentence_length()
        self.dict_path = os.path.join(self.bert_dir,'vocab.txt')
        
        if bool(self.bert_enabled):
            loaded_dic = Dictionary(dict_path =self.dict_path)
            self.embedding = Embedding(loaded_dic,self.max_sequence_length)
        else:
            self.embedding = Embedding(self.get_dictionary(self.datas.values()),self.max_sequence_length)
        print('loading word embedding...')
        self.embedding.get_embedding(dataset_name = self.dataset_name, fname=opt.wordvec_path)
        self.opt_callback(opt) 
    
    def opt_callback(self,opt):
        opt.nb_classes = self.nb_classes            
        opt.embedding_size = self.embedding.lookup_table.shape[1]        
        opt.max_sequence_length= self.max_sequence_length
        
        opt.lookup_table = self.embedding.lookup_table      
    
    def get_dictionary(self, corpuses= None,dataset="",fresh=True):
        pkl_name="temp/"+self.dataset_name+".alphabet.pkl"
        if os.path.exists(pkl_name) and not fresh:
            return pickle.load(open(pkl_name,"rb"))
        dictionary = Dictionary(start_feature_id = 0)
        dictionary.add('[UNK]')  
#        alphabet.add('END') 
        for corpus in corpuses:
            for sentence in corpus['X']:    
                tokens = sentence.lower().split()
                for token in set(tokens):
                    dictionary.add(token)
        print("dictionary size = {}".format(len(dictionary.keys())))
        if not os.path.exists("temp"):
            os.mkdir("temp")
        pickle.dump(dictionary,open(pkl_name,"wb"))
        return dictionary   
    
    def prepare_data(self,seqs):
        lengths = [len(seq) for seq in seqs]
        n_samples = len(seqs)
        max_len = self.max_sequence_length
    
        x = np.zeros((n_samples, max_len)).astype('int32')
        x_mask = np.zeros((n_samples, max_len)).astype('float')
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.0
         # print( x, x_mask)
        return x, x_mask
    
    def get_train(self,shuffle = True,iterable=True,max_sequence_length=0):
        x = self.datas['train']['X']
        x = [self.embedding.text_to_sequence(sent) for sent in x]
        y = to_categorical(np.asarray(self.datas['train']['y']))
        
        data = (x,y)
        if max_sequence_length == 0:
            max_sequence_length = self.max_sequence_length
        if iterable:
            return BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=max_sequence_length) 
        else: 
            if self.bert_enabled:
                x,x_mask = to_array(x,maxlen = self.max_sequence_length, use_mask = True)
                return [x,x_mask],y
            else:
                x = to_array(x,maxlen = self.max_sequence_length, use_mask = False)
                return x,y
        
        
    def get_test(self,shuffle = True,iterable=True,max_sequence_length=0):
        x = self.datas['test']['X']
        x = [self.embedding.text_to_sequence(sent) for sent in x]
        y = to_categorical(np.asarray(self.datas['test']['y']))
        data = (x,y)
        if max_sequence_length == 0:
            max_sequence_length = self.max_sequence_length
        if iterable:
            return BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=max_sequence_length) 
        else: 
            if self.bert_enabled:
                x,x_mask = to_array(x,maxlen = self.max_sequence_length, use_mask = True)
                return [x,x_mask],y
            else:
                x = to_array(x,maxlen = self.max_sequence_length, use_mask = False)
                return x,y
        
        
    def get_val(self,shuffle = True,iterable=True,max_sequence_length=0):
        x = self.datas['dev']['X']
        x = [self.embedding.text_to_sequence(sent) for sent in x]
        y = to_categorical(np.asarray(self.datas['dev']['y']))
        data = (x,y)
        if max_sequence_length == 0:
            max_sequence_length = self.max_sequence_length
        if iterable:
            return BucketIterator(data,batch_size=self.batch_size,shuffle=True,max_sequence_length=max_sequence_length) 
        else: 
            if self.bert_enabled:
                x,x_mask = to_array(x,maxlen = self.max_sequence_length, use_mask = True)
                return [x,x_mask],y
            else:
                x = to_array(x,maxlen = self.max_sequence_length, use_mask = False)
                return x,y
        
        
    def preprocess(self, data):
        data['X'] = self.preprocessor.run_seq(data['X'],output_type = 'string')
        data['y'] = data['y']
        return(data)
        
    def get_max_sentence_length(self):

        samples = self.datas['train']['X'] + self.datas['dev']['X'] + \
                self.datas['test']['X']
        max_sentence_length = 0
        if self.bert_enabled:
            self.max_sequence_length = 512
        else:
            for sample in samples:
                sample_length = len(sample.split())
                if max_sentence_length < sample_length:
                    max_sentence_length = sample_length
            self.max_sequence_length = max_sentence_length

    def get_word_embedding(self, path_to_vec,orthonormalized=True):
        samples = self.data['train']['X'] + self.data['dev']['X'] + \
                self.data['test']['X']

        id2word, word2id = create_dictionary(samples, threshold=0)

        word_vec = get_wordvec(path_to_vec, word2id,orthonormalized=orthonormalized)
        wvec_dim = len(word_vec[next(iter(word_vec))])

        #stores the value of theta for each word
        word_complex_phase = set_wordphase(word2id)

        idfs = np.array([0.5]*(len(word2id)+1))
        counter = Counter([word for sen in self.data['train']['X'] for word in sen])
        for index, word in enumerate(id2word):
            if word in counter:
                idfs[index+1] = counter[word]
        idfs[0] = 100
        idfs = np.log(np.sum(idfs)/idfs)
        params = {'word2id':word2id, 'word_vec':word_vec, 'wvec_dim':wvec_dim,'word_complex_phase':word_complex_phase,'id2word':id2word,"id2idf":idfs}
        self.embedding_params = params
        return params

    def create_batch(self, embedding_params, batch_size = -1):
        embed = {'train': {}, 'dev': {}, 'test': {}}
        for key in self.data:
            embed[key] = {'X':[],'y':[]}
            logging.info('Computing embedding for {0}'.format(key))
            sorted_data = sorted(zip(self.data[key]['X'],
                                     self.data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.data[key]['X'], self.data[key]['y'] = map(list, zip(*sorted_data))
            bsize = batch_size
            if (batch_size == -1):
                bsize = len(self.data[key]['y'])
            for ii in range(0, len(self.data[key]['y']), bsize):
                batch = self.data[key]['X'][ii:ii + bsize]
                embeddings = get_index_batch(embedding_params, batch)
                # print(embeddings)
                embed[key]['X'].append(embeddings)
                # print(self.sst_data[key]['y'][ii:ii + batch_size])
                embed[key]['y'].append(self.data[key]['y'][ii:ii + bsize])
            # sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
            # print(sst_embed[key]['y'])
            embed[key]['y'] = np.array(embed[key]['y'])
            # print(sst_embed[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))
        return embed



    def get_processed_data(self):

        train_test_val = self.create_batch(self.embedding_params)
        training_data = train_test_val['train']
        test_data = train_test_val['test']
        validation_data = train_test_val['dev']

        train_x, train_y = data_gen(training_data, self.max_sentence_length)
        test_x, test_y = data_gen(test_data, self.max_sentence_length)
        val_x, val_y = data_gen(validation_data, self.max_sentence_length)

        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)
        val_y = to_categorical(val_y)
        return (train_x, train_y),(test_x, test_y),(val_x, val_y)



class TRECDataReader(DataReader):
    def __init__(self, task_dir_path, preprocessor, seed=1111):
        self.seed = seed
        train = self.loadFile(os.path.join(task_dir_path, 'train_5500.label'))
        train, dev = self.train_dev_split(train, train_dev_ratio = 1/9)
        test = self.loadFile(os.path.join(task_dir_path, 'TREC_10.label'))
        nb_classes = 6
        super(TRECDataReader,self).__init__(train, dev, test, nb_classes, preprocessor)
        self.nb_classes = nb_classes

    def train_dev_split(self, samples, train_dev_ratio = 1/9):
        X_train, X_dev, y_train, y_dev = train_test_split(samples['X'], samples['y'], test_size=train_dev_ratio, random_state=self.seed)
        train = {'X': X_train, 'y':y_train}
        dev = {'X': X_dev, 'y':y_dev}
        return train, dev

    def loadFile(self, fpath):
        trec_data = {'X': [], 'y': []}
        tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
                'HUM': 3, 'LOC': 4, 'NUM': 5}
        with io.open(fpath, 'r', encoding='latin-1') as f:
            for line in f:
                target, sample = line.strip().split(':', 1)
                sample = sample.split(' ', 1)[1]
                assert target in tgt2idx, target
                trec_data['X'].append(sample)
                trec_data['y'].append(tgt2idx[target])
        return trec_data



class SSTDataReader(DataReader):
    def __init__(self, task_dir_path, opt, nclasses = 2, seed = 1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'

        train = self.loadFile(os.path.join(task_dir_path, self.task_name,'sentiment-train'))
        dev = self.loadFile(os.path.join(task_dir_path, self.task_name, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_dir_path, self.task_name, 'sentiment-test'))
#        super().__init__(train, dev, test, nclasses)
        super(SSTDataReader,self).__init__(train, dev, test, nclasses, opt)
        self.nb_classes = nclasses

    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.nclasses == 2:
                    sample = line.strip().split('\t')
                    sst_data['y'].append(int(sample[1]))
                    sst_data['X'].append(sample[0])
                elif self.nclasses == 5:
                    sample = line.strip().split(' ', 1)
                    sst_data['y'].append(int(sample[0]))
                    sst_data['X'].append(sample[1])
        assert max(sst_data['y']) == self.nclasses - 1
        return sst_data

class BinaryClassificationDataReader(DataReader):
    def __init__(self, pos, neg, opt, seed=1111):
        self.seed = seed
        self.samples, self.labels = pos + neg, [1] * len(pos) + [0] * len(neg)
        train, test, dev = self.train_test_dev_split(0.1,1.0/9)
        nb_classes = 2
        super(BinaryClassificationDataReader,self).__init__(train, test, dev, opt, nb_classes)
        self.nb_classes = nb_classes

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line for line in f.read().splitlines()]

    def train_test_dev_split(self, train_test_ratio = 0.1,train_dev_ratio = 1/9):
        X_train, X_test, y_train, y_test = train_test_split(self.samples, self.labels, test_size=train_test_ratio, random_state=self.seed)

        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=train_dev_ratio, random_state=self.seed)
        train = {'X': X_train, 'y':y_train}
        test = {'X': X_test, 'y':y_test}
        dev = {'X': X_dev, 'y':y_dev}
        return train, test, dev


class CRDataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, opt, seed=1111):
        # logging.debug('***** Transfer task : CR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'custrev.pos'))
        neg = self.loadFile(os.path.join(task_path, 'custrev.neg'))
        super(CRDataReader,self).__init__(pos, neg, opt, seed)


class MRDataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, opt, seed=1111):
        # logging.debug('***** Transfer task : MR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'rt-polarity.pos'))
        neg = self.loadFile(os.path.join(task_path, 'rt-polarity.neg'))
        super(MRDataReader,self).__init__(pos, neg, opt, seed)


class SUBJDataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, opt, seed=1111):
        # logging.debug('***** Transfer task : SUBJ *****\n\n')
        obj = self.loadFile(os.path.join(task_path, 'subj.objective'))
        subj = self.loadFile(os.path.join(task_path, 'subj.subjective'))
        super(SUBJDataReader,self).__init__(obj, subj, opt, seed)


class MPQADataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, opt, seed=1111):
        # logging.debug('***** Transfer task : MPQA *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'mpqa.pos'))
        neg = self.loadFile(os.path.join(task_path, 'mpqa.neg'))
        super(MPQADataReader,self).__init__(pos, neg, opt, seed)

#def data_reader_initialize(reader_type, datasets_dir):
#    dir_path = os.path.join(datasets_dir, reader_type)
#    if(reader_type == 'CR'):
#        return(CRDataReader(dir_path))
#    if(reader_type == 'MR'):
#        return(MRDataReader(dir_path))
#    if(reader_type == 'SUBJ'):
#        return(SUBJDataReader(dir_path))
#    if(reader_type == 'MPQA'):
#        return(MPQADataReader(dir_path))
#    if(reader_type == 'SST_2'):
#        dir_path = os.path.join(datasets_dir, 'SST')
#        return(SSTDataReader(dir_path, nclasses = 2))
#    if(reader_type == 'SST_5'):
#        dir_path = os.path.join(datasets_dir, 'SST')
#        return(SSTDataReader(dir_path, nclasses = 5))
#    if(reader_type == 'TREC'):
#        return(TRECDataReader(dir_path))
