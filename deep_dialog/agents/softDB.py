#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
'''

import numpy as np

class SoftDB:
    def _inform(self, probs):
        '''
        确认不需要再进行request时，直接将每行感兴趣的程度降序排列输出
        :param probs:
        :return: 根据感兴趣程度降序排列的行号
        '''
        return np.argsort(probs)[::-1].tolist()

    def _check_db(self):
        '''
        induce distribution over DB based on current beliefs over slots
        '''
        probs = {}
        p_s = np.zeros((self.state['database'].N, len(self.state['database'].slots))).astype('float32')  # (N,|S|)
        for i,s in enumerate(self.state['database'].slots):
            p = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum() # slot[i]下的每个value的概率，不含UNK
            n = self.state['database'].inv_counts[s] # slot[i]下每个value的的出现频数
            p_unk = float(n[-1])/self.state['database'].N # slot[i]下UNK的频率
            p_tilde = p*(1.-p_unk)
            p_tilde = np.concatenate([p_tilde,np.asarray([p_unk])]) # slot[i]下的每个value的概率，含UNK
            p_s[:,i] = p_tilde[self.state['database'].table[:,i]] / n[self.state['database'].table[:,i]]
        p_db = np.sum(np.log(p_s), axis=1)
        p_db = np.exp(p_db - np.min(p_db))
        p_db = p_db/p_db.sum()   # 归一化
        return p_db  # N


