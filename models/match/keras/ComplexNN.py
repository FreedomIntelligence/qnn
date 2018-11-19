
# -*- coding: utf-8 -*-
from models.match.keras.SiameseNetwork import SiameseNetwork
from models.representation.keras.ComplexNN import ComplexNN as representation_model

class ComplexNN(SiameseNetwork):
    
    def initialize(self):
        super(ComplexNN, self).initialize()
        self.representation_model = representation_model(self.opt)
    def __init__(self,opt):
        super(ComplexNN, self).__init__(opt) 

