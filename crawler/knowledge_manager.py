import numpy as np
class KnowledgeManager():
    def __init__(self):
        self.ids = {}
        self.ns = {}
        self.non0 = {}
        self.unks = {}
        self.inv_index = {}
        self.inv_counts = {}
        self.table = np.zeros().astype('int16')
        self.counts = np.zeros().astype('float32')

