from .preprocessor import Preprocess

def setup(opt):
    #default
    remove_punctuation = False
    stem = False
    remove_stopwords = False
    
    #read from configuration file
    if 'remove_punctuation' in opt.__dict__:
        remove_punctuation = opt.remove_punctuation
    if 'stem' in opt.__dict__:
        stem = opt.stem    
    if 'remove_stopwords':
        remove_stopwords = opt.remove_stopwords
    preprocessor = Preprocess(remove_punctuation, stem, remove_stopwords)
    return preprocessor