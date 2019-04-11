# -*- coding: utf-8 -*-
import traceback,sys,os




def setup(opt):
    config_adapter(opt)
    model = import_object(opt.network_type, opt.__dict__)        
    return model

def import_class(import_str):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.insert(0,dirname)
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))


def import_object(import_str, *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)



def config_adapter(conf):
    conf.text1_maxlen = conf.max_sequence_length
    conf.embed = conf.lookup_table
    conf.vocab_size = len(conf.alphabet)
    conf.hidden_sizes = [20, 1]
    conf.num_layers = 2
    conf.embed_size = conf.embedding_size
    conf.bin_num =1
    conf.target_mode = "ranking"
    
    
    




if __name__ == "__main__":
    from params import Params
    params = Params()
    config_file = 'config/qalocal.ini'    # define dataset in the config
    params.parse_config(config_file)
    params.network_type = "anmm.ANMM"

    
    from dataset import qa
    reader = qa.setup(params)
    from models.match import keras as models
    model = models.setup(params)
    