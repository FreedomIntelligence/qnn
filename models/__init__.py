# -*- coding: utf-8 -*-


#from models.match.keras.RealNN import RealNN
#from models.match.keras.QDNN import QDNN
#from models.match.keras.ComplexNN import ComplexNN
##from .QDNNAblation import QDNNAblation
#from models.match.keras.LocalMixtureNN import LocalMixtureNN

def setup(opt):
    
    if opt.language == 'keras':
        if opt.dataset_type == 'qa':
            from models.match.keras.RealNN import RealNN
            from models.match.keras.QDNN import QDNN
            from models.match.keras.ComplexNN import ComplexNN
            from models.match.keras.LocalMixtureNN import LocalMixtureNN
            
        elif opt.dataset_type == 'classification':
            from models.classification.keras.RealNN import RealNN
            from models.classification.keras.QDNN import QDNN
            from models.classification.keras.ComplexNN import ComplexNN
#            from models.classification.keras.QDNNAblation import QDNNAblation
            from models.classification.keras.LocalMixtureNN import LocalMixtureNN
    
    elif opt.language == 'tensorflow':
        if opt.dataset_type == 'qa':
            from models.match.tensorflow.RealNN import RealNN
            from models.match.tensorflow.QDNN import QDNN
            from models.match.tensorflow.ComplexNN import ComplexNN
            from models.match.tensorflow.LocalMixtureNN import LocalMixtureNN
            
            
        elif opt.dataset_type == 'classification':
            from models.classification.tensorflow.RealNN import RealNN
            from models.classification.tensorflow.QDNN import QDNN
            from models.classification.tensorflow.ComplexNN import ComplexNN
            from models.classification.tensorflow.QDNNAblation import QDNNAblation
            from models.classification.tensorflow.LocalMixtureNN import LocalMixtureNN
    
    elif opt.language == 'torch':
        if opt.dataset_type == 'qa':
#            from models.match.pytorch.RealNN import RealNN
#            from models.match.pytorch.QDNN import QDNN
#            from models.match.pytorch.ComplexNN import ComplexNN
#            from models.match.pytorch.LocalMixtureNN import LocalMixtureNN
            print('None')
            
        elif opt.dataset_type == 'classification':
#            from models.classification.pytorch.RealNN import RealNN
            from models.classification.pytorch.QDNN import QDNN
#            from models.classification.pytorch.ComplexNN import ComplexNN
#            from models.classification.pytorch.QDNNAblation import QDNNAblation
#            from models.classification.pytorch.LocalMixtureNN import LocalMixtureNN
    
    
    print("network type: " + opt.network_type)
    if opt.network_type == "real":
        model = RealNN(opt)
    elif opt.network_type == "qdnn":
        model = QDNN(opt)
    elif opt.network_type == "complex":
        model = ComplexNN(opt)
    elif opt.network_type == "local_mixture":
        model = LocalMixtureNN(opt)
        
    elif opt.network_type == "bert":
        from models.representation.keras.BertFasttext import BERTFastext
        model = BERTFastext(opt)
        
    elif opt.network_type == "ablation":
        print("run ablation")
        model = QDNNAblation(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
