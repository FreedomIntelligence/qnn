
# -*- coding: utf-8 -*-
from .classification import *
from .qa import *
def setup(opt):
    if opt.dataset_type == 'qa':
        reader = qa.setup(opt)
    elif opt.dataset_type == 'classification':
        reader = classification.setup(opt)
    else: # By default the type is classification
        reader = classification.setup(opt)

    return reader

    
    