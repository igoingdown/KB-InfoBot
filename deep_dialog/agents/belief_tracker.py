#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
'''

import nltk
import numpy as np
import time

from collections import Counter, defaultdict
from deep_dialog.tools import to_tokens

UPD = 10

class BeliefTracker:
    def _search(self,w_t,s_t):
        #w_t = to_tokens(w)
        return float(sum([ww in s_t for ww in w_t]))/len(w_t)

    def _search_slots(self, s_t):
        '''
        在用户输入中查找出现的slot名称，这里已经转为了token操作
        :param s_t: 用户输入的token化结果
        :return: 匹配统计结果
        '''
        matches = {}
        for slot,slot_t in self.state['database'].slot_tokens.iteritems():
            m = self._search(slot_t,s_t)
            if m>0.: 
                matches[slot] = m
        return matches

    def _search_values(self, s_t):
        '''
        在用户输入中查找出现的slot value，这里已经转为了token操作
        :param s_t: 用户输入的token化结果
        :return: 匹配统计结果
        '''
        print('-' * 100 + "\nsearching values: ")
        for v in s_t:
            print(v.encode("utf8") if v is not None and type(v) == unicode else v)
        matches = {}
        for slot in self.state['database'].slots:
            matches[slot] = defaultdict(float)
            for ss in s_t:
                if ss in self.movie_dict.tokens[slot]:
                    for vi in self.movie_dict.tokens[slot][ss]:
                        matches[slot][vi] += 1.
            for vi,f in matches[slot].iteritems():
                val = self.movie_dict.dict[slot][vi]
                # TODO: 中文版一定要删掉nltk的东西
                print(nltk.word_tokenize(val))
                matches[slot][vi] = f/len(nltk.word_tokenize(val))
        print('-' * 100)
        return matches

    def _update_state(self, user_utterance, upd=UPD, verbose=False):
        '''
        根据用户输入，暴力查找用户输入中的关键字(token)，就地改变BT的状态
        :param user_utterance: 用户输入
        :param upd: 当前训练次数(模型更新次数)
        :param verbose: 是否启用唠叨模式
        :return: None
        '''
        prev_act, prev_slot = self.state['prevact'].split('@')

        s_t = to_tokens(user_utterance)
        slot_match = self._search_slots(s_t) # search slots
        val_match = self._search_values(s_t) # search values

        for slot, values in val_match.iteritems():
            requested = (prev_act=='request') and (prev_slot==slot)
            matched = (slot in slot_match)
            if not values:
                if requested: # asked for value but did not get it，就不再关心这个slot了！
                    self.state['database'].delete_slot(slot)
                    self.state['num_requests'][slot] = 1000
                    self.state['dont_care'].add(slot)
            else:
                for y, match in values.iteritems():
                    #y = self.movie_dict.dict[slot].index(val)
                    if verbose:
                        v = self.movie_dict.dict[slot][y]
                        print 'Detected %s' %v.encode("utf8") if v is not None and type(v) == unicode else v, ' update = ', match
                    if matched and requested:
                        alpha = upd*(match + 1. + slot_match[slot])
                    elif matched and not requested:
                        alpha = upd*(match + slot_match[slot])
                    elif not matched and requested:
                        alpha = upd*(match + 1.)
                    else:
                        alpha = upd*match
                    self.state['inform_slots'][slot][y] += alpha
                    # TODO: inform_slots到底是记录什么的？为什么要乘上10？
                self.state['slot_tracker'].add(slot)

    def _init_beliefs(self):
        '''
        初始化BT，用Database的先验分布(均匀分布)对每个slot的BT进行初始化
        :return: 初始化过的BT
        '''
        beliefs = {s:np.copy(self.state['database'].priors[s]) 
                for s in self.state['database'].slots}
        return beliefs
