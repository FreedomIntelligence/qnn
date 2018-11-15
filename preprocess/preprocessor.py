# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
import re, string
import jieba
import sys

class Preprocess(object):
    stopwords_set = set(stopwords.words('english'))
    stemmer=SnowballStemmer('english')
    valid_lang = ['en', 'cn'] 
    def __init__(self, opt = None):
        
        #default settings
        self.remove_punctuation = False
        self.stem = False
        
        
        # word segmentation
        self.word_seg_enable = True
        self.word_seg_lang = 'en'
        
        # word stemming
        self.word_stem_enable = True
        
        # word lowercase
        self.word_lower_enable = True
        
        
        # stop word removal
        self.stopword_remove_enable = True
        
        # punctuation removal
        self.punct_remove_enable = True
        self.preprocess_silent_mode = True
#        self._doc_filter_config = { 'enable': True, 'min_len': 0, 'max_len': six.MAXSIZE }
        
       
#        self._word_filter_config = { 'enable': True, 'stop_words': nltk_stopwords.words('english'),
#                                     'min_freq': 1, 'max_freq': six.MAXSIZE, 'words_useless': None }
#        self._word_index_config = { 'word_dict': None }
#        
#        self._doc_filter_config.update(doc_filter_config)
#        self._word_stem_config.update(word_stem_config)
#        self._word_lower_config.update(word_lower_config)
#        self._word_filter_config.update(word_filter_config)
#        self._word_index_config.update(word_index_config)
#
#        self._word_dict = self._word_index_config['word_dict']
#        self._words_stats = dict()
        if not opt is None:
            for key,value in opt.__dict__.items():
                self.__setattr__(key,value)  
        
        self._word_seg_config = { 'enable':self.word_seg_enable ,'lang': self.word_seg_lang }
        self._word_stem_config = { 'enable': self.word_stem_enable}
        self._word_lower_config = { 'enable': self.word_lower_enable }
        self._stopword_remove_config = { 'enable': self.stopword_remove_enable}
        self._punct_remove_config = {'enable': self.punct_remove_enable}
        
#    def run2(self, file_path):
#        print('load...')
#        dids, docs = Preprocess.load(file_path)
#
#        if self._word_seg_config['enable']:
#            print('word_seg...')
#            docs = Preprocess.word_seg(docs, self._word_seg_config)
#
#        if self._doc_filter_config['enable']:
#            print('doc_filter...')
#            dids, docs = Preprocess.doc_filter(dids, docs, self._doc_filter_config)
#
#        if self._word_stem_config['enable']:
#            print('word_stem...')
#            docs = Preprocess.word_stem(docs)
#
#        if self._word_lower_config['enable']:
#            print('word_lower...')
#            docs = Preprocess.word_lower(docs)
#
#        self._words_stats = Preprocess.cal_words_stat(docs)
#
#        if self._word_filter_config['enable']:
#            print('word_filter...')
#            docs, self._words_useless = Preprocess.word_filter(docs, self._word_filter_config, self._words_stats)
#
#        print('word_index...')
#        docs, self._word_dict = Preprocess.word_index(docs, self._word_index_config)
#
#        return dids, docs
    

    def run(self,sentence, output_type ='list'):
        
        if self._punct_remove_config['enable']:
            if not self.preprocess_silent_mode:
                print('punct_remove...')
            sentence = Preprocess.punct_remove(sentence)
        
        if self._word_seg_config['enable']:
            if not self.preprocess_silent_mode:
                print('word_seg...')
            sentence = Preprocess.word_seg(sentence, self._word_seg_config)
        else:
            return sentence
        
        if self._word_stem_config['enable']:
            if not self.preprocess_silent_mode:
                print('word_stem...')
            sentence = Preprocess.word_stem(sentence)
        if self._word_lower_config['enable']:
            if not self.preprocess_silent_mode:
                print('word_lower...')
            sentence = Preprocess.word_lower(sentence)
            
        if self._stopword_remove_config['enable']:
            if not self.preprocess_silent_mode:
                print('stopword_remove...')
            sentence = Preprocess.stopword_remove(sentence)
        
        if output_type == 'string':
            sentence = " ".join(sentence)

#        if self._doc_filter_config['enable']:
#        if self.remove_punctuation:
#            sentence = re.sub('[%s]' % re.escape(string.punctuation), ' ', sentence)
#        sentence = [w for w in word_tokenize(sentence.lower())]
#        if self.stem:
#            sentence = [stemmer.stem(w) for w in sentence]
#        if self.remove_stopwords:
#            sentence = [w for w in sentence if w not in stopwords_set]
       
        return sentence
    
                
    def run_seq(self,sentences, output_type = 'list'):
        
        output = []
#        print('preprocess begins.')
        for sentence in sentences:
            output.append(self.run(sentence, output_type = output_type))
#        print('preprocess ends.')
        return output
    
    @staticmethod
    def stopword_remove(sentence):
        sentence = [w for w in sentence if w not in Preprocess.stopwords_set]
        return sentence
    
    @staticmethod
    def punct_remove(sentence):
        sentence = re.sub('[{}]'.format(re.escape(string.punctuation)), ' ', sentence)
        return sentence
    
    @staticmethod
    def word_lower(sentence):
        sentence = [w.lower() for w in sentence]
        return sentence
    
    @staticmethod
    def word_seg_en(sentence):
        sentence= word_tokenize(sentence) 
        # show the progress of word segmentation with tqdm
        '''docs_seg = []
        print('docs size', len(docs))
        for i in tqdm(range(len(docs))):
            docs_seg.append(word_tokenize(docs[i]))'''
        return sentence
    
    @staticmethod
    def word_seg_cn(sentence):
        sentence = list(jieba.cut(sentence))
#        docs = [list(jieba.cut(sent)) for sent in docs]
        return sentence

    @staticmethod
    def word_seg(sentence, config):
        assert config['lang'].lower() in Preprocess.valid_lang, 'Wrong language type: {}'.format(config['lang'])
        sentence = getattr(Preprocess, '{}_{}'.format(sys._getframe().f_code.co_name, config['lang']))(sentence)
        return sentence
    
    @staticmethod
    def word_stem(sentence):
        sentence = [Preprocess.stemmer.stem(w) for w in sentence]
        return sentence
    
   

    
if __name__ == '__main__':
    a = ['Today is a good day!','it is a good day today']
    preprocessor = Preprocess()
    print(preprocessor.run_seq(a, output_type = 'string'))