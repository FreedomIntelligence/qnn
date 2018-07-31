from .RealNN import RealNN
from .QDNN import QDNN
from .ComplexNN import ComplexNN
def setup(opt):
    if opt.network_type == "real":
        model = RealNN(opt)
    elif opt.network_type == "qdnn":
        model = QDNN(opt)
    elif opt.network_type == "complex":
        model = ComplexNN(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model