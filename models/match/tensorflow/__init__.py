# -*- coding: utf-8 -*-
from .QARNN import QARNN
def setup(opt):
    print("network type: " + opt.network_type)
    network_type = opt.network_type.strip().lower()
    if network_type == "qarnn":
        model = QARNN(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
