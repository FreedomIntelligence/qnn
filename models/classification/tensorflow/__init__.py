# -*- coding: utf-8 -*-
from .QA_CNN import QA_quantum
def setup(opt):
    print("network type: " + opt.network_type)
    network_type = opt.network_type.strip().lower()
    if network_type == "qacnn":
        model = QA_quantum(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model