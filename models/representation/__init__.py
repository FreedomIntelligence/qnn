from models.representation.keras.RealNN import RealNN
from models.representation.keras.QDNN import QDNN
from models.representation.keras.ComplexNN import ComplexNN
from models.representation.keras.QDNNAblation import QDNNAblation
from models.representation.keras.LocalMixtureNN import LocalMixtureNN
from models.representation.keras.BertFasttext import BERTFastext
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
    elif opt.network_type == "bert":
        model = BERTFastext(opt)
    elif opt.network_type == "ablation":
        print("run ablation")
        model = QDNNAblation(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
