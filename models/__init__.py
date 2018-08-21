from models.RealNN import RealNN
from models.QDNN import QDNN
from models.ComplexNN import ComplexNN
from models.QDNNAblation import QDNNAblation
from models.LocalMixtureNN import LocalMixtureNN

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
