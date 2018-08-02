from __future__ import division
import os
import io
import logging
import numpy as np
import data as data
from sklearn.model_selection import train_test_split

class DataReader(object):
    def __init__(self, train, dev, test, nb_classes):
        self.data = {'train': train, 'dev': dev, 'test': test}
        self.nb_classes = nb_classes
        self.max_sentence_length = self.get_max_sentence_length()

    def get_max_sentence_length(self):
        samples = self.data['train']['X'] + self.data['dev']['X'] + \
                self.data['test']['X']
        max_sentence_length = 0
        for sample in samples:
            sample_length = len(sample)
            if max_sentence_length < sample_length:
                max_sentence_length = sample_length

        return max_sentence_length

    def get_word_embedding(self, path_to_vec,orthonormalized=True):
        samples = self.data['train']['X'] + self.data['dev']['X'] + \
                self.data['test']['X']

        id2word, word2id = data.create_dictionary(samples, threshold=0)
        word_vec = data.get_wordvec(path_to_vec, word2id,orthonormalized=orthonormalized)
        wvec_dim = len(word_vec[next(iter(word_vec))])

        #stores the value of theta for each word
        word_complex_phase = data.set_wordphase(word2id)

        params = {'word2id':word2id, 'word_vec':word_vec, 'wvec_dim':wvec_dim,'word_complex_phase':word_complex_phase,'id2word':id2word}

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
                embeddings = data.get_index_batch(embedding_params, batch)
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


class TRECDataReader(DataReader):
    def __init__(self, task_dir_path, seed=1111):
        self.seed = seed
        train = self.loadFile(os.path.join(task_dir_path, 'train_5500.label'))
        train, dev = self.train_dev_split(train, train_dev_ratio = 1/9)
        test = self.loadFile(os.path.join(task_dir_path, 'TREC_10.label'))
        nb_classes = 6
        super(TRECDataReader,self).__init__(train, dev, test, nb_classes)

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
                sample = sample.split(' ', 1)[1].split()
                assert target in tgt2idx, target
                trec_data['X'].append(sample)
                trec_data['y'].append(tgt2idx[target])
        return trec_data



class SSTDataReader(DataReader):
    def __init__(self, task_dir_path, nclasses = 2, seed = 1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = 'Binary' if self.nclasses == 2 else 'Fine-Grained'

        train = self.loadFile(os.path.join(task_dir_path, self.task_name,'sentiment-train'))
        dev = self.loadFile(os.path.join(task_dir_path, self.task_name, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_dir_path, self.task_name, 'sentiment-test'))
#        super().__init__(train, dev, test, nclasses)
        super(SSTDataReader,self).__init__(train, dev, test, nclasses)

    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.nclasses == 2:
                    sample = line.strip().split('\t')
                    sst_data['y'].append(int(sample[1]))
                    sst_data['X'].append(sample[0].split())
                elif self.nclasses == 5:
                    sample = line.strip().split(' ', 1)
                    sst_data['y'].append(int(sample[0]))
                    sst_data['X'].append(sample[1].split())
        assert max(sst_data['y']) == self.nclasses - 1
        return sst_data

class BinaryClassificationDataReader(DataReader):
    def __init__(self, pos, neg, seed=1111):
        self.seed = seed
        self.samples, self.labels = pos + neg, [1] * len(pos) + [0] * len(neg)
        train, test, dev = self.train_test_dev_split(0.1,1.0/9)
        nb_classes = 2
        super(BinaryClassificationDataReader,self).__init__(train, test, dev, nb_classes)

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    def train_test_dev_split(self, train_test_ratio = 0.1,train_dev_ratio = 1/9):
        X_train, X_test, y_train, y_test = train_test_split(self.samples, self.labels, test_size=train_test_ratio, random_state=self.seed)

        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=train_dev_ratio, random_state=self.seed)
        train = {'X': X_train, 'y':y_train}
        test = {'X': X_test, 'y':y_test}
        dev = {'X': X_dev, 'y':y_dev}
        return train, test, dev


class CRDataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, seed=1111):
        # logging.debug('***** Transfer task : CR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'custrev.pos'))
        neg = self.loadFile(os.path.join(task_path, 'custrev.neg'))
        super(CRDataReader,self).__init__(pos, neg, seed)


class MRDataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, seed=1111):
        # logging.debug('***** Transfer task : MR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'rt-polarity.pos'))
        neg = self.loadFile(os.path.join(task_path, 'rt-polarity.neg'))
        super(MRDataReader,self).__init__(pos, neg, seed)


class SUBJDataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, seed=1111):
        # logging.debug('***** Transfer task : SUBJ *****\n\n')
        obj = self.loadFile(os.path.join(task_path, 'subj.objective'))
        subj = self.loadFile(os.path.join(task_path, 'subj.subjective'))
        super(SUBJDataReader,self).__init__(obj, subj, seed)


class MPQADataReader(BinaryClassificationDataReader):
    def __init__(self, task_path, seed=1111):
        # logging.debug('***** Transfer task : MPQA *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'mpqa.pos'))
        neg = self.loadFile(os.path.join(task_path, 'mpqa.neg'))
        super(MPQADataReader,self).__init__(pos, neg, seed)

def data_reader_initialize(reader_type, datasets_dir):
    dir_path = os.path.join(datasets_dir, reader_type)
    if(reader_type == 'CR'):
        return(CRDataReader(dir_path))
    if(reader_type == 'MR'):
        return(MRDataReader(dir_path))
    if(reader_type == 'SUBJ'):
        return(SUBJDataReader(dir_path))
    if(reader_type == 'MPQA'):
        return(MPQADataReader(dir_path))
    if(reader_type == 'SST_2'):
        dir_path = os.path.join(datasets_dir, 'SST')
        return(SSTDataReader(dir_path, nclasses = 2))
    if(reader_type == 'SST_5'):
        dir_path = os.path.join(datasets_dir, 'SST')
        return(SSTDataReader(dir_path, nclasses = 5))
    if(reader_type == 'TREC'):
        return(TRECDataReader(dir_path))
