# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
import re, string
stemmer=SnowballStemmer('english')
class Preprocess(object):

    def __init__(self,
             remove_punctuation = False,
             stem = False,
             remove_stopwords = False
                 ):
        self.remove_punctuation = remove_punctuation
        self.stem = stem
        self.remove_stopwords = remove_stopwords
    

    def run(self,sentence):
    
        if self.remove_punctuation:
            sentence = re.sub('[%s]' % re.escape(string.punctuation), ' ', sentence)
            sentence = [w for w  in word_tokenize(sentence.lower())]
        if self.stem:
            sentence = [stemmer.stem(w) for w  in sentence]
        if self.remove_stopwords:
            sentence = [w for w in sentence if w not in stopwords_set]
        return " ".join(sentence)
    
    def run_seq(self,sentences):
        output = []
        for sentence in sentences:
            if self.remove_punctuation:
                sentence = re.sub('[%s]' % re.escape(string.punctuation), ' ', sentence)
                sentence = [w for w  in word_tokenize(sentence.lower())]
            if self.stem:
                sentence = [stemmer.stem(w) for w in sentence]
            if self.remove_stopwords:
                sentence = [w for w in sentence if w not in stopwords_set]
            output.append(sentence)
        return output
    
if __name__ == '__main__':
    a = ['today is a good day','it is a good day today']
    
    preprocessor = Preprocess(stem = True)
    print(preprocessor.run(a[0].split()))