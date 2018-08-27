#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
a rule-based user simulator

'''

import argparse, json, time
import random
import copy
import nltk

from deep_dialog import dialog_config

DOMAIN_NAME = 'movie'

def weighted_choice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices,weights):
        if upto + w >= r:
            return c
        upto += w
    assert False, "shouldn't get here"

class RuleSimulator:
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, 
            start_set=None, max_turn=20, nlg=None, err_prob=0., db=None, 
            dk_prob=0., sub_prob=0., max_first_turn=5):
        '''

        :param movie_dict:
        :param act_set:
        :param slot_set:
        :param start_set: ??
        :param max_turn: max turn in one dialog episode, 20 by default
        :param nlg: natural language generator
        :param err_prob: the probability of the user simulator corrupting a slot value
        :param db: database
        :param dk_prob: the probability that user simulator does not know a slot value
        :param sub_prob: the probability that user simulator substitutes a slot value
        :param max_first_turn:
        '''
        self.max_turn = dialog_config.MAX_TURN
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        self.nlg = nlg
        self.err_prob = err_prob
        self.database = db
        self.dk_prob = dk_prob
        self.sub_prob = sub_prob
        self.max_first_turn = max_first_turn

    def _sample_action(self):
        '''
        randomly sample a start state
        :return: 当前episode是否结束标志，对话状态
        '''
        self.state = {}
        self.state['diaact'] = random.choice(dialog_config.start_dia_acts.keys())
        self.state['turn'] = 0
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['prev_diaact'] = 'UNK'
        
        if (len(self.goal['inform_slots']) + len(self.goal['request_slots'])) > 0:
            # 从goal的inform_slots中抽取一部分作为当前state的inform_slots
            if len(self.goal['inform_slots']) > 0:
                care_about = [s for s,v in self.goal['inform_slots'].iteritems() if v is not None]
                known_slots = random.sample(care_about, 
                        random.randint(1,min(self.max_first_turn,len(care_about))))
                for s in known_slots:
                    self.state['inform_slots'][s] = self.goal['inform_slots'][s]
            # 从goal的request_slots中抽取一个作为当期state的request_slots
            if len(self.goal['request_slots']) > 0:
                request_slot = random.choice(self.goal['request_slots'].keys())
                self.state['request_slots'][request_slot] = 'UNK'
        
        if (self.state['diaact'] in ['thanks','closing']): episode_over = True
        else: episode_over = False

        # 生成state的inform_slots_noisy，加入噪声
        if not episode_over:
            self.corrupt()
        
        sent = self.nlg.generate(self.state['diaact'],self.state['request_slots'],
                self.state['inform_slots_noisy']) if self.nlg is not None else ''
        self.state['nl_sentence'] = sent
        self.state['episode_over'] = episode_over
        self.state['reward'] = 0
        return episode_over, self.state

    def _sample_goal(self):
        '''
        sample a goal. goal的target就是target record的index.
        goal的known slots就是该target record的非Missing Value(UNK)的slot及其slot value组成的KV对。
        :return: 修改user simulator对象的goal即可，不必返回
        '''
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
            # known_slots 是database中target record所有已知的所有slot及值

            care_about = random.sample(known_slots, int(self.dk_prob*len(known_slots)))
            # 从数据库中已知slot的value的known_slots中随机抽取一部分座位用户感兴趣的care_about

            # 填充target的相关slot到goal的inform_slots中，用户关心的slot填充的一定是正确的，但是其余的都不填充
            for i,s in enumerate(self.database.slots):
                if s not in dialog_config.inform_slots: continue
                val = self.database.tuples[self.goal['target']][i]
                if s in care_about and val!='UNK':
                    self.goal['inform_slots'][s] = val
                else:
                    self.goal['inform_slots'][s] = None

            # 防止goal的inform_slots的V全都是None，因为care_about可能为空。
            # 这时候可以适当放宽上面的要求，把非care_about的slot设为正确值即可。
            if all([v==None for v in self.goal['inform_slots'].values()]):
                while True:
                    s = random.choice(self.goal['inform_slots'].keys())
                    i = self.database.slots.index(s)
                    val = self.database.tuples[self.goal['target']][i]
                    if val!='UNK':
                        self.goal['inform_slots'][s] = val
                        break

    def print_goal(self):
        '''
        打印用户的goal
        :return:
        '''
        print 'User target = ', ', '.join(['%s:%s' %(s,v) for s,v in \
                zip(['movie']+self.database.slots, \
                [self.database.labels[self.goal['target']]] + \
                self.database.tuples[self.goal['target']])])
        print 'User information = ', ', '.join(['%s:%s' %(s,v) for s,v in \
                self.goal['inform_slots'].iteritems() if v is not None]), '\n'

    def initialize_episode(self):
        '''
        initialization，先采样一条记录用户期待的答案，然后随机采样生成用户的初始输入(action)
        :return: 返回用户开始的action
        '''
        self._sample_goal()
        
        # first action
        episode_over, user_action = self._sample_action()
        assert (episode_over != 1),' but we just started'
        return user_action

    def next(self, state):
        '''
        用户根据系统状态决定什么样的输入，输入之后修改对话状态
        :param state: sys_action
        :return: 新的对话状态，即用户对话状态
        '''
        self.state['turn'] += 1
        reward = 0
        episode_over = False
        self.state['prev_diaact'] = self.state['diaact']
        self.state['inform_slots'].clear()
        self.state['request_slots'].clear()
        self.state['inform_slots_noisy'].clear()
        
        if (self.max_turn > 0 and self.state['turn'] >= self.max_turn):
            reward = dialog_config.FAILED_DIALOG_REWARD
            episode_over = True
            self.state['diaact'] = 'deny'
        else:
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
            elif act == 'request':
                slot = state['request_slots'].keys()[0]
                if slot in self.goal['inform_slots']:
                    self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                else:
                    self.state['inform_slots'][slot] = None
                self.state['diaact'] = 'inform'
                reward = dialog_config.PER_TURN_REWARD

        if not episode_over:
            self.corrupt()

        sent = self.nlg.generate(self.state['diaact'],self.state['request_slots'],
                self.state['inform_slots_noisy']) if self.nlg is not None else ''
        self.state['nl_sentence'] = sent
        self.state['episode_over'] = episode_over
        self.state['reward'] = reward

        return self.state, episode_over, reward

    def corrupt(self):
        '''
        用户可能会犯浑！用户的记忆是不牢靠的，以一定概率修改用户的先验知识
        :return:
        '''
        self.state['inform_slots_noisy'] = {}
        for slot in self.state['inform_slots'].keys():
            if self.state['inform_slots'][slot]==None:
                self.state['inform_slots_noisy'][slot] = None
            else:
                prob_sub = random.random()
                if prob_sub < self.sub_prob:
                    # substitute value，偷天换日，随机抽样该slot下的一个值将目标值替换掉
                    self.state['inform_slots_noisy'][slot] = \
                            random.choice(self.movie_dict.dict[slot])
                else:
                    self.state['inform_slots_noisy'][slot] = self.state['inform_slots'][slot]
                prob_err = random.random()
                if prob_err < self.err_prob:
                    # corrupt value，对于数字进行污染
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

    def stateVector(self, action):
        '''
        user state representation
        :param action:
        :return:
        '''
        vec = [0]*(len(self.act_set.dict) + len(self.slot_set.slot_ids)*2)

        if action['diaact'] in self.act_set.dict.keys(): vec[self.act_set.dict[action['diaact']]] = 1
        for slot in action['slots'].keys():
            slot_id = self.slot_set.slot_ids[slot] * 2 + len(self.act_set.dict)
            slot_id += 1
            if action['slots'][slot] == 'UNK': vec[slot_id] =1

        return vec

    def print_state(self, action):
        '''
        print the state
        :param action: 用户采取的action，即用户输入
        :return:
        '''
        stateStr = 'Turn %d user action: %s, history slots: %s, inform_slots: %s, request slots: %s, rest_slots: %s' % (action['turn'], action['diaact'], action['history_slots'], action['inform_slots'], action['request_slots'], action['rest_slots'])
        print stateStr



def main(params):
    user_sim = RuleSimulator()
    user_sim.init()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print 'User Simulator Parameters: '
    print json.dumps(params, indent=2)

    main(params)
