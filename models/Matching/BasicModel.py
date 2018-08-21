# -*- coding: utf-8 -*-

class BasicModel(object):
    def __init__(self,opt):        
        self.opt=opt
        self.initialize()
        self.model = self.build()
    
    def initialize(self):
        pass
    
    def build(self):
        return None
    
        
    def getModel(self):
        return self.model
        