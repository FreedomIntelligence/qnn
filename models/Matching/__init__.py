# -*- coding: utf-8 -*-

#from .RealNN import RealNN
#from .QDNN import QDNN
#from .ComplexNN import ComplexNN
#from .QDNNAblation import QDNNAblation
from .LocalMixtureNN import LocalMixtureNN

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
        
    elif opt.network_type == "ablation":
        print("run ablation")
        model = QDNNAblation(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
