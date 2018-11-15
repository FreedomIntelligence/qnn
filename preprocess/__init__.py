from .preprocessor import Preprocess

def setup(opt):
    #default
    preprocessor = Preprocess(opt)
    return preprocessor