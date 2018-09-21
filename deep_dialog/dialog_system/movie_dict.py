#!/usr/bin/python
# -*- coding: utf-8 -*-

'''

'''

import cPickle as pickle
import copy
import nltk
import string

from collections import defaultdict
from deep_dialog.tools import to_tokens

class MovieDict:
    def __init__(self, path):
        self.load_dict(path)
        self.count_values()
        self._build_token_index()
    
    def load_dict(self, path):
        dict_data = pickle.load(open(path, 'rb'))
        self.dict = copy.deepcopy(dict_data)
        # 构造一份自己的dict
        # print("-" * 200 + "movie dict: {}\n".format(self.dict) + "-" * 200)

    def count_values(self):
        self.lengths = {}
        for k,v in self.dict.iteritems():
            self.lengths[k] = len(v)

    def _build_token_index(self):
        self.tokens = {}
        # tokens的结构:{slot_name:{word_token: [slot value IDs]}}
        for slot,vals in self.dict.iteritems():
            # print "db slot: {}\nslot values: {}".format(slot.encode("utf-8"), vals)
            self.tokens[slot] = defaultdict(list)
            for vi,vv in enumerate(vals):
                w_v = to_tokens(vv)
                for w in w_v: self.tokens[slot][w].append(vi)
