#!/usr/bin/python
# -*- coding: utf-8 -*-

from deep_dialog import tools

def standardize(arr):
    return arr

def calc_entropies(state, q, db):
    '''
    SL中计算熵的方式，跟RL中不一样！
    :param state:
    :param q: table probability, (N,)
    :param db: database
    :return: 每个slot的熵
    '''
    entropies = {}
    for s,c in state.iteritems():
        if s not in db.slots:
            entropies[s] = 0.
        else:
            p = (db.ids[s]*q).sum(axis=1)
            u = db.priors[s]*q[db.unks[s]].sum()
            c_tilde = p+u
            c_tilde = c_tilde/c_tilde.sum()
            entropies[s] = tools.entropy_p(c_tilde)
    return entropies
