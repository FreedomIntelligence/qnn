# -*- coding: utf-8 -*-
from models.match.keras.SiameseNetwork import SiameseNetwork
from models.representation.keras.RealNN import RealNN as representation_model

class RealNN(SiameseNetwork):

    def initialize(self):
        super(RealNN, self).initialize()
        self.representation_model = representation_model(self.opt)
    def __init__(self,opt):
        super(RealNN, self).__init__(opt) 



