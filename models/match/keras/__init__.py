# -*- coding: utf-8 -*-

from models.match.keras.RealNN import RealNN
from models.match.keras.QDNN import QDNN
from models.match.keras.ComplexNN import ComplexNN
#from .QDNNAblation import QDNNAblation
from models.match.keras.LocalMixtureNN import LocalMixtureNN

def setup(opt):
    print("network type: " + opt.network_type)
    if opt.network_type == "real":
        model = RealNN(opt)
    elif opt.network_type == "qdnn":
        model = QDNN(opt)
    elif opt.network_type == "complex":
        model = ComplexNN(opt)
    elif opt.network_type == "local_mixture":
        model = LocalMixtureNN(opt)        
#    elif opt.network_type == "ablation":
#        print("run ablation")
#        model = QDNNAblation(opt)
    else:
        from models.match.keras import matchzoo
        model = matchzoo.setup(opt)
    return model
