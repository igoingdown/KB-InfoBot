#!/usr/bin/python
# -*- coding: utf-8 -*-

'''

Class for database
'''

import csv
import io
import numpy as np
import nltk
import time

from collections import defaultdict
from deep_dialog import dialog_config
from deep_dialog.tools import to_tokens

class Database:
    def __init__(self, path, dicts, name=''):
        self.path = path
        self.name = name
        self._load_db(path)
        self._shuffle()
        self._build_inv_index(dicts)
        self._build_table(dicts)
        self._get_priors()
        self._prepare_for_entropy(dicts)
        self._prepare_for_search()

    def _load_db(self, path):
        '''
        读取电影数据集，保存slots(除电影名之外的6种属性)、labels(每部电影的电影名列表)和tuples(每部电影的其余6种属性的列表)
        :param path: 数据集的path
        :return: None
        '''
        with open(path, 'r') as f:
            self.slots = [x.decode("utf8") for x in f.readline().rstrip().split('\t')[1:]]
            self.labels = []
            self.tuples = []
            for line in f:
                tuple = [x.decode("utf8") for x in line.rstrip().split('\t')]
                self.labels.append(tuple[0])
                self.tuples.append(tuple[1:])
        # try:
        #     fi = io.open(path,'r')
        #     self.slots = fi.readline().rstrip().split('\t')[1:]
        #     tupl = [line.rstrip().split('\t') for line in fi]
        #     self.labels = [t[0] for t in tupl]
        #     self.tuples = [t[1:] for t in tupl]
        #     fi.close()
        # except UnicodeDecodeError:
        #     fi = open(path,'r')
        #     self.slots = fi.readline().rstrip().split('\t')[1:]
        #     tupl = [line.rstrip().split('\t') for line in fi]
        #
        #     self.labels = [t[0] for t in tupl]
        #     self.tuples = [t[1:] for t in tupl]
        #     fi.close()

        self.N = len(self.tuples)

    def _shuffle(self):
        # match slot order to config
        # for v in dialog_config.inform_slots:
        #     print(type(v))
        #     print v.encode("utf8") if v is not None and type(v) == unicode else v
        # print('-' * 200)
        # for v in self.slots:
        #     print(type(v))
        #     print v.encode("utf8") if v is not None and type(v) == unicode else v

        index = [self.slots.index(s) for s in dialog_config.inform_slots]
        self.slots = [self.slots[ii] for ii in index]
        self.tuples = [[row[ii] for ii in index] for row in self.tuples]

    def lookup(self, query, match_unk=True):
        def _iseq(t1, t2):
            for i in range(len(t1)):
                if t1[i]!=t2[i] and t1[i]!='UNK' and t2[i]!='UNK':
                    return False
            return True
        col_idx = [ii for ii,vv in enumerate(query) if vv is not None]
        c_db = [[row[ii] for ii in col_idx] for row in self.tuples]
        c_q = [query[ii] for ii in col_idx]
        if match_unk: row_match_idx = [ii for ii,ll in enumerate(c_db) if _iseq(ll,c_q)]
        else: row_match_idx = [ii for ii,ll in enumerate(c_db) if ll==c_q]
        results = [self.tuples[ii] for ii in row_match_idx]
        return results, row_match_idx

    def delete_slot(self, slot):
        try:
            slot_index = self.slots.index(slot)
        except ValueError:
            print 'Slot not found!!!'
            return
        for row in self.tuples: del row[slot_index]
        self.table = np.delete(self.table, slot_index, axis=1)
        self.counts = np.delete(self.counts, slot_index, axis=1)
        del self.slots[slot_index]

    def _build_inv_index(self, dicts):
        self.inv_index = {}
        self.inv_counts = {}
        for i,slot in enumerate(self.slots):
            V = dicts.lengths[slot]
            self.inv_index[slot] = defaultdict(list)
            self.inv_counts[slot] = np.zeros((V+1,)).astype('float32')
            values = [t[i] for t in self.tuples]
            # 每条记录的第i个slot值列表
            # print "dicts.dict[{}]: {}".format(slot.encode("utf8"), dicts.dict[slot])
            for j,v in enumerate(values):
                v_id = dicts.dict[slot].index(v) if v!='UNK' else V
                # print "v: {}, dicts.dict[slot].index(v): {}".format(v, v_id)
                self.inv_index[slot][v].append(j)
                self.inv_counts[slot][v_id] += 1
        # raise Exception("hello world")

    def _build_table(self, dicts):
        self.table = np.zeros((len(self.tuples),len(self.slots))).astype('int16')
        self.counts = np.zeros((len(self.tuples),len(self.slots))).astype('float32')
        for i,t in enumerate(self.tuples):
            for j,v in enumerate(t):
                s = self.slots[j]
                self.table[i,j] = dicts.dict[s].index(v) if v!='UNK' else dicts.lengths[s]
                self.counts[i,j] = self.inv_counts[s][self.table[i,j]]

    def _get_priors(self):
        self.priors = {slot:self.inv_counts[slot][:-1]/self.inv_counts[slot][:-1].sum() \
                for slot in self.slots}

    def _prepare_for_entropy(self, dicts):
        '''

        :param dicts: movie dict
        :return:
        '''
        self.ids = {}
        self.ns = {}
        self.non0 = {}
        self.unks = {}
        for i,s in enumerate(self.slots):
            V = dicts.lengths[s]
            table_column_i = self.table[:,i]
            self.unks[s] = np.where(table_column_i==V)[0]
            self.ids[s] = (np.mgrid[:self.priors[s].shape[0],:self.N]==table_column_i)[0]
            # ids记录每个slot的value在各行出现的次数，是否出现，shape为|V| * N，第i条记录的第j个slot的value为 v_k 时，ids[k][i] = True.
            # grids = np.mgrid[:self.priors[s].shape[0],:self.N]
            # print("-" * 200 + "\ngrid0 shape:{} * {}, grid1 shape: {} * {}\ntable_column_i: {}\n".format(len(grids[0]), len(grids[0][0]), len(grids[1]), len(grids[1][0]), len(table_column_i)) + "-" * 200)
            # print("-" * 200 + "\n第 {} 个 slot：slot {} 的ids shape: {} * {}\n".format(i, s, len(self.ids[s]), len(self.ids[s][0])) + "-" * 200)
            self.ns[s] = self.ids[s].sum(axis=1)
            self.non0[s] = np.nonzero(self.ns[s])[0]

    def _prepare_for_search(self):
        self.slot_tokens = {}
        for slot in self.slots:
            self.slot_tokens[slot] = to_tokens(slot)
