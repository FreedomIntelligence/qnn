# -*- coding: utf-8 -*-

from models.match.RealNN import RealNN
from models.match.QDNN import QDNN
from models.match.ComplexNN import ComplexNN
#from .QDNNAblation import QDNNAblation
from models.match.LocalMixtureNN import LocalMixtureNN

from models.match.SiameseNetwork import SiameseNetwork

def setup(opt):
    print("matching network type: Siamese Network with " + opt.network_type)

    model = SiameseNetwork(opt)

    
    
#    if opt.network_type == "real":
#        model = RealNN(opt)
#    elif opt.network_type == "qdnn":
#        model = QDNN(opt)
#    elif opt.network_type == "complex":
#        model = ComplexNN(opt)
#    elif opt.network_type == "local_mixture":
#        model = LocalMixtureNN(opt)        
##    elif opt.network_type == "ablation":
##        print("run ablation")
##        model = QDNNAblation(opt)
#    else:
#        raise Exception("model not supported: {}".format(opt.network_type))
    return model
