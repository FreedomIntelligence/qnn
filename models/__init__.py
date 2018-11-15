# -*- coding: utf-8 -*-


#from models.match.keras.RealNN import RealNN
#from models.match.keras.QDNN import QDNN
#from models.match.keras.ComplexNN import ComplexNN
##from .QDNNAblation import QDNNAblation
#from models.match.keras.LocalMixtureNN import LocalMixtureNN

def setup(opt):
    if opt.dataset_type == 'qa':
        from models.match.keras.RealNN import RealNN
        from models.match.keras.QDNN import QDNN
        from models.match.keras.ComplexNN import ComplexNN
        from models.match.keras.LocalMixtureNN import LocalMixtureNN
        
        
    elif opt.dataset_type == 'classification':
        from models.representation.RealNN import RealNN
        from models.representation.QDNN import QDNN
        from models.representation.ComplexNN import ComplexNN
        from models.representation.QDNNAblation import QDNNAblation
        from models.representation.LocalMixtureNN import LocalMixtureNN
        from models.representation.BertFasttext import BERTFastext
    
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
        model = BERTFastext(opt)
        
    elif opt.network_type == "ablation":
        print("run ablation")
        model = QDNNAblation(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
