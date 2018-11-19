# -*- coding: utf-8 -*-

from models.match.keras.SiameseNetwork import SiameseNetwork
from models.representation.keras.LocalMixtureNN import  LocalMixtureNN as representation_model

class LocalMixtureNN(SiameseNetwork):

    def initialize(self):
        super(LocalMixtureNN, self).initialize()
        self.representation_model = representation_model(self.opt)
    def __init__(self,opt):
        super(LocalMixtureNN, self).__init__(opt) 
