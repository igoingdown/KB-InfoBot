#!/usr/bin/python
# -*- coding: utf-8 -*-

'''

a rule-based user simulator

'''

import argparse, json, time
import random
import copy
import nltk
import cPickle as pkl
import sys, locale
from deep_dialog import dialog_config
from deep_dialog.tools import to_tokens
from collections import defaultdict

# 要改为中文
# DOMAIN_NAME = 'movie'
# GENERIC = ['I dont know', 'I cannot remember', 'I am not sure']
DOMAIN_NAME = u'影片'
GENERIC = [u'我不清楚', u'我不太记得了', u'我不确定']

def weighted_choice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices,weights):
        if upto + w >= r:
            return c
        upto += w
    # 改为中文
    # assert False, "shouldnt get here"
    assert False, "出问题啦"

class CmdUser:
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, 
            start_set=None, max_turn=20, err_prob=0., db=None, 
            dk_prob=0., sub_prob=0., max_first_turn=5,
            fdict_path=None):
        self.max_turn = dialog_config.MAX_TURN
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        self.err_prob = err_prob
        self.database = db
        self.dk_prob = dk_prob
        self.sub_prob = sub_prob
        self.max_first_turn = max_first_turn
        self.grams = self._load_vocab(fdict_path)
        self.N = 2

    def _load_vocab(self, path):
        if path is None: return set()
        else: return pkl.load(open(path,'rb'))

    def _vocab_search(self, text):
        '''
        按照2-Gram的方式查看是否用户输入中是否存在关键字
        :param text: 用户输入的自然语言句子
        :return: 是否包含关键字，或者说用户输入的问题是否有意义
        '''
        tokens = to_tokens(text)
        for i in range(len(tokens)):
            for t in range(self.N):
                if i-t<0: continue
                ngram = u''.join(tokens[i-t:i+1])
                if ngram in self.grams: 
                    return True
        return False

    ''' show target entity and known slots (corrupted) to user
        and get NL input '''
    def prompt_input(self, agent_act, turn):
        print ''
        print '轮次: {}'.format(turn)
        print '系统动作: ', agent_act
        print '目标 ', DOMAIN_NAME.encode("utf8"), ': ', self.database.labels[self.goal['target']].encode("utf8")
        print 'known slots: ', ' '.join(
                ['%s={ %s }' %(k.encode("utf8") if k is not None and type(k) == unicode else k,
                               u' , '.join(vv for vv in v).encode("utf8"))
                    for k,v in self.state['inform_slots_noisy'].iteritems()])

        inp = raw_input('你的输入: ').decode("utf8")
        if not self._vocab_search(inp): return random.choice(GENERIC)
        else: return inp

    ''' display agent results at end of dialog '''
    def display_results(self, ranks, reward, turns):
        print '\n系统结果: ', u'， '.join([self.database.labels[ii] for ii in ranks[:5]]).encode("utf8")
        print '目标影片排名:{0}'.format(ranks.index(self.goal['target']) + 1)
        if reward > 0: print '对话成功!'
        else: print '对话失败'
        print '对话轮数:{0}'.format(turns)

    ''' randomly sample a start state '''
    def _sample_action(self):
        self.state = {}
        
        self.state['diaact'] = ''
        self.state['turn'] = 0
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['prev_diaact'] = 'UNK'

        self.corrupt()
        # 改到中文
        #sent = self.prompt_input('Hi! I am Info-Bot. I can help you search for movies if you tell me their attributes!', 0).lower()
        sent = self.prompt_input('你好，向我提问关于电影的东西吧！',0).lower()
        sent = sent.dencode("utf8") if sent is not None and type(sent) == str else sent
        if sent==u'quit': episode_over=True
        else: episode_over=False
        
        self.state['nl_sentence'] = sent
        self.state['episode_over'] = episode_over
        self.state['reward'] = 0
        self.state['goal'] = self.goal['target']

        return episode_over, self.state

    ''' sample a goal '''
    def _sample_goal(self):
        if self.start_set is not None:
            self.goal = random.choice(self.start_set)  # sample user's goal from the dataset
        else:
            # sample a DB record as target
            self.goal = {}
            self.goal['request_slots'] = {}
            self.goal['request_slots'][DOMAIN_NAME] = 'UNK'
            self.goal['target'] = random.randint(0,self.database.N-1)
            self.goal['inform_slots'] = {}
            known_slots = [s for i,s in enumerate(dialog_config.inform_slots) 
                    if self.database.tuples[self.goal['target']][i]!='UNK']
            care_about = random.sample(known_slots, int(self.dk_prob*len(known_slots)))
            for i,s in enumerate(self.database.slots):
                if s not in dialog_config.inform_slots: continue
                val = self.database.tuples[self.goal['target']][i]
                if s in care_about and val!='UNK':
                    self.goal['inform_slots'][s] = val
                else:
                    self.goal['inform_slots'][s] = None
            if all([v==None for v in self.goal['inform_slots'].values()]):
                while True:
                    s = random.choice(self.goal['inform_slots'].keys())
                    i = self.database.slots.index(s)
                    val = self.database.tuples[self.goal['target']][i]
                    if val!='UNK':
                        self.goal['inform_slots'][s] = val
                        break

    def print_goal(self):
        # 改到中文
        print '用户目标 = ', ', '.join(['%s:%s' %(s.encode("utf8") if s is not None and type(s) == unicode else s,
                                              v.encode("utf8") if v is not None and type(v) == unicode else v) \
                                    for s,v in zip(['影片']+self.database.slots,
                                                   [self.database.labels[self.goal['target']]] +
                                                   self.database.tuples[self.goal['target']])])
        print '用户输入信息 = ', ', '.join(['%s:%s' %(s.encode("utf8") if s is not None and type(s) == unicode else s,
                                              v.encode("utf8") if v is not None and type(v) == unicode else v)
                                                for s,v in self.goal['inform_slots'].iteritems() if v is not None]), '\n'

    ''' initialization '''
    def initialize_episode(self):
        self._sample_goal()
        
        # first action
        episode_over, user_action = self._sample_action()
        assert (episode_over != 1),' but we just started'
        return user_action

    ''' update state: state is sys_action '''
    def next(self, state):
        self.state['turn'] += 1
        reward = 0
        episode_over = False
        self.state['prev_diaact'] = self.state['diaact']
        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()
        
        act = state['diaact']
        if act == 'inform':
            episode_over = True
            goal_rank = state['target'].index(self.goal['target'])
            if goal_rank < dialog_config.SUCCESS_MAX_RANK:
                reward = dialog_config.SUCCESS_DIALOG_REWARD*\
                        (1.-float(goal_rank)/dialog_config.SUCCESS_MAX_RANK)
                self.state['diaact'] = 'thanks'
            else:
                reward = dialog_config.FAILED_DIALOG_REWARD
                self.state['diaact'] = 'deny'
            self.display_results(state['target'], reward, self.state['turn'])
        else:
            slot = state['request_slots'].keys()[0]
            agent_act = act + ' ' + slot
            sent = self.prompt_input(agent_act, self.state['turn']).lower()
            if sent=='quit' or self.state['turn'] >= self.max_turn: episode_over=True
            reward = 0
            self.state['nl_sentence'] = sent

        self.state['episode_over'] = episode_over
        self.state['reward'] = reward

        return self.state, episode_over, 0

    ''' user may make mistakes '''
    def corrupt(self):
        self.state['inform_slots_noisy'] = {}
        for slot in self.goal['inform_slots'].keys():
            self.state['inform_slots_noisy'][slot] = set()
            if self.goal['inform_slots'][slot] is not None:
                cset = set([self.goal['inform_slots'][slot]])
                prob_sub = random.random()
                if prob_sub < self.sub_prob: # substitute value
                    cset.add(random.choice(self.movie_dict.dict[slot]))
                for item in cset:
                    prob_err = random.random()
                    if prob_err < self.err_prob: # corrupt value
                        self.state['inform_slots_noisy'][slot].update(
                                self._corrupt_value(item))
                    #else:
                    #    self.state['inform_slots_noisy'][slot].add(item)
                    self.state['inform_slots_noisy'][slot].add(item)

    def _corrupt_value(self, val):
        def _is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        def _is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        # 中文不用nltk
        # tokens = nltk.word_tokenize(val)
        tokens = to_tokens(val)
        if len(tokens)>1: 
            tokens.pop(random.randrange(len(tokens)))
            out = set([' '.join(tokens)])
        else:
            t = tokens[0]
            out = set()
            if _is_int(t):
                pert = round(random.gauss(0,0.5))
                if pert>0: out.add('%d' %(int(t)+pert))
                out.add(t)
            elif _is_float(t):
                pert = random.gauss(0,0.5)
                if pert>0.05: out.add('%.1f' %(float(t)+pert))
                out.add(t)
            else:
                out.add(t)
        return out

    ''' user may make mistakes
    def corrupt(self):
        self.state['inform_slots_noisy'] = {}
        for slot in self.goal['inform_slots'].keys():
            if self.goal['inform_slots'][slot]==None:
                self.state['inform_slots_noisy'][slot] = None
            else:
                prob_sub = random.random()
                if prob_sub < self.sub_prob: # substitute value
                    self.state['inform_slots_noisy'][slot] = \
                            random.choice(self.movie_dict.dict[slot])
                else:
                    self.state['inform_slots_noisy'][slot] = self.goal['inform_slots'][slot]
                prob_err = random.random()
                if prob_err < self.err_prob: # corrupt value
                    self.state['inform_slots_noisy'][slot] = \
                            self._corrupt_value(self.state['inform_slots_noisy'][slot])

    def _corrupt_value(self, val):
        def _is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        def _is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        tokens = nltk.word_tokenize(val)
        if len(tokens)>1: tokens.pop(random.randrange(len(tokens)))
        out = []
        for t in tokens:
            if _is_int(t):
                out.append(str(int(random.gauss(int(t),0.5))))
            elif _is_float(t):
                out.append('%.1f' %random.gauss(float(t),0.5))
            else:
                out.append(t)
        return ' '.join([o for o in out])
    '''

    ''' user state representation '''
    def stateVector(self, action):
        vec = [0]*(len(self.act_set.dict) + len(self.slot_set.slot_ids)*2)

        if action['diaact'] in self.act_set.dict.keys(): vec[self.act_set.dict[action['diaact']]] = 1
        for slot in action['slots'].keys():
            slot_id = self.slot_set.slot_ids[slot] * 2 + len(self.act_set.dict)
            slot_id += 1
            if action['slots'][slot] == 'UNK': vec[slot_id] =1

        return vec

    ''' print the state '''
    def print_state(self, action):
        stateStr = 'Turn %d user action: %s, history slots: %s, inform_slots: %s, request slots: %s, rest_slots: %s' % (action['turn'], action['diaact'], action['history_slots'], action['inform_slots'], action['request_slots'], action['rest_slots'])
        print stateStr



def main(params):
    pass
    # user_sim = RuleSimulator()
    # user_sim.init()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print 'User Simulator Parameters: '
    print json.dumps(params, indent=2)

    main(params)
