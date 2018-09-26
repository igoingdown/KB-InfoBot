#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
end2end agent' implementation
'''

import numpy as np
import cPickle as pkl

from deep_dialog import dialog_config, tools
from collections import Counter, defaultdict, deque
from agent_lu_rl import E2ERLAgent, aggregate_rewards
from belief_tracker import BeliefTracker
from softDB import SoftDB
from feature_extractor import FeatureExtractor
from utils import *

import operator
import random
import math
import copy
import re
import nltk
import time

# params
DISPF = 1
SAVEF = 100
ANNEAL = 800

class AgentE2ERLAllAct(E2ERLAgent,SoftDB,BeliefTracker):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, db=None, corpus=None,
            train=True, _reload=False, n_hid=100, batch=128, ment=0., input_type='full', upd=10,
            sl='e2e', rl='e2e', pol_start=600, lr=0.005, N=1, tr=2.0, ts=0.5, max_req=2, frac=0.5, 
            name=None):
        '''
        构造end2end的Agent
        :param movie_dict:
        :param act_set:
        :param slot_set:
        :param db: database
        :param corpus:
        :param train: 指定 train 还是 evaluate
        :param _reload: 在测试阶段需要使用reload重载训练的模型
        :param n_hid: hidden size
        :param batch: batch size
        :param ment: Entropy regularization parameter，RL正则化项loss的权重，不需要正则化项时设为0，这也是默认设置！
        :param input_type: 确定policy network的特征输入模式，可以是entropy或者full,entropy时使用计算熵的方式对输入特征进行降维
        :param upd: Update count for bayesian belief tracking
        :param sl: 指定 supervised learning 应用于哪个网络, 可以用于belief tracking，policy network或者两者都使用(e2e，默认设定)
        :param rl: 指定reinforcement learning 应用于哪个网络，可以用于belief tracking, policy network或者两者都用(e2e，默认设定)
        :param pol_start: 将RL应用于policy network的iteration下限
        :param lr:learning rate, 对RL和SL有不同的设置，而且SL的learning rate不变，而RL的learning rate逐渐变小
        :param N: featN, N-gram's N, used in simple rule feature extraction，一般是2
        :param tr: database entropy's threshold to inform，用于基于规则的action选择
        :param ts: slot entropy's threshold to request，用于基于规则的action，如果有任意一个slot不够确定，就继续提问
        :param max_req: Maximum requests allowed for per slot
        :param frac: Ratio to initial slot entropy, 一个slot的entropy如果低于这个下限，这个slot就再也不会被问到(request)
        :param name:
        '''
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.database = db
        self.max_turn = dialog_config.MAX_TURN
        self.training = train
        self.feat_extractor = FeatureExtractor(corpus,self.database.path,N=N)
        out_size = len(dialog_config.inform_slots)+1
        in_size = len(self.feat_extractor.grams) + len(dialog_config.inform_slots)
        slot_sizes = [self.movie_dict.lengths[s] for s in dialog_config.inform_slots]
        self._init_model(in_size, out_size, slot_sizes, self.database,
                n_hid=n_hid, learning_rate_sl=lr, batch_size=batch, ment=ment, input_type=input_type,
                sl=sl, rl=rl)
        self._name = name
        if _reload: self.load_model(dialog_config.MODEL_PATH+self._name)
        if train: self.save_model(dialog_config.MODEL_PATH+self._name)
        self._init_experience_pool(batch)
        self.episode_count = 0

        # 记录recent的reward，loss等监控模型的训练进程, 在_print_progress函数中有用到。
        self.recent_rewards = deque([], 1000)
        self.recent_successes = deque([], 1000)
        self.recent_turns = deque([], 1000)
        self.recent_loss = deque([], 10)

        self.discount = 0.99
        self.num_updates = 0
        self.pol_start = pol_start
        self.tr = tr
        self.ts = ts
        self.max_req = max_req
        self.frac = frac
        self.upd = upd

    def _print_progress(self,loss,te,*args):
        self.recent_loss.append(loss)
        avg_ret = float(sum(self.recent_rewards))/len(self.recent_rewards)
        avg_turn = float(sum(self.recent_turns))/len(self.recent_turns)
        avg_loss = float(sum(self.recent_loss))/len(self.recent_loss)
        n_suc, n_fail, n_inc, tot = 0, 0, 0, 0
        for s in self.recent_successes:
            if s==-1: n_fail += 1
            elif s==0: n_inc += 1
            else: n_suc += 1
            tot += 1
        if len(args)>0:
            print 'Update %d. Avg turns = %.2f . Avg Reward = %.2f . Success Rate = %.2f . Fail Rate = %.2f . Incomplete Rate = %.2f . Loss = %.3f . Time = %.2f' % \
                    (self.num_updates, avg_turn, avg_ret,
                    float(n_suc)/tot, float(n_fail)/tot, float(n_inc)/tot, avg_loss, te)
            #print 'kl loss = {}'.format(args[0])
            #print 'x_loss = {}'.format(args[1])
        else:
            print 'Update %d. Avg turns = %.2f . Avg Reward = %.2f . Success Rate = %.2f . Fail Rate = %.2f . Incomplete Rate = %.2f . Loss = %.3f . Time = %.2f' % \
                    (self.num_updates, avg_turn, avg_ret,
                    float(n_suc)/tot, float(n_fail)/tot, float(n_inc)/tot, avg_loss, te)

    def initialize_episode(self):
        '''
        每次开启一次新的对话(episode)之前，在这里更新agent的参数
        在user simulator初始化之后，根据对话状态(user_action)初始化agent
        :return: None
        '''
        self.episode_count += 1
        if self.training and self.episode_count%self.batch_size==0:
            self.num_updates += 1
            if self.num_updates>self.pol_start and self.num_updates%ANNEAL==0: self.anneal_lr()
            tst = time.time()
            if self.num_updates < self.pol_start:
                # 初始条件下使用SL方式更新参数
                all_loss = self.update(regime='SL')
                loss = all_loss[0]
                # loss是BT的loss与action的loss的和
                kl_loss = all_loss[1:len(dialog_config.inform_slots)+1]
                # kl_loss:每个slot的模型预测p和手工计算的p_target的KL散度
                x_loss = all_loss[len(dialog_config.inform_slots)+1:]
                # x_loss每个slot的模型预测q和手工计算q_target的交叉熵
                t_elap = time.time() - tst
                if self.num_updates%DISPF==0: self._print_progress(loss, t_elap, kl_loss, x_loss)
            else:
                # 开始使用RL的方式更新参数
                loss = self.update(regime='RL')
                t_elap = time.time() - tst
                if self.num_updates%DISPF==0: self._print_progress(loss, t_elap)
            if self.num_updates%SAVEF==0: self.save_model(dialog_config.MODEL_PATH+self._name)

        self.state = {}
        # #解决dumps的bug
        # fields = dir(self.database)
        # for f in fields:
        #     print "{}: {}".format(f, type(getattr(self.database, f)))
        #     if not f.startswith("__"):
        #         try:
        #             pkl.dumps(getattr(self.database, f), -1)
        #         except Exception as e:
        #             print("{}: dump failed!, exception: {}".format(f, e))
        # database_dumps = pkl.dumps(self.database, -1)
        # print(type(database_dumps), database_dumps)
        # database_loads = pkl.loads(database_dumps)
        # print(type(database_loads), database_loads)

        self.state['database'] = pkl.loads(pkl.dumps(self.database,-1))
        self.state['prevact'] = 'begin@begin'
        self.state['inform_slots'] = self._init_beliefs()
        self.state['turn'] = 0
        self.state['num_requests'] = {s:0 for s in self.state['database'].slots}
        self.state['slot_tracker'] = set()
        self.state['dont_care'] = set()
        p_db_i = (1./self.state['database'].N)*np.ones((self.state['database'].N,))
        self.state['init_entropy'] = calc_entropies(self.state['inform_slots'], p_db_i, 
                self.state['database'])
        self.state['inputs'] = []
        self.state['actions'] = []
        self.state['rewards'] = []
        self.state['indices'] = []   # 保存inform action之后的概率前几名
        self.state['ptargets'] = []
        self.state['phitargets'] = []
        self.state['hid_state'] = [np.zeros((1,self.r_hid)).astype('float32') \
                for s in dialog_config.inform_slots]
        self.state['pol_state'] = np.zeros((1,self.n_hid)).astype('float32')

    def next(self, user_action, verbose=False):
        '''
        get next action based on rules
        :param user_action: 用户输入之后，新的state
        :param verbose: 是否打印模型运行过程产生的log，是否开启唠叨模式
        :return: 返回action的dict里面有其他的参数，包括diaact,request_slots,target,p和q等
        '''
        self.state['turn'] += 1

        p_vector = np.zeros((self.in_size,)).astype('float32')   # （|Grams|+|Slots|, )
        p_vector[:self.feat_extractor.n] = self.feat_extractor.featurize(user_action['nl_sentence'])
        if self.state['turn']>1:
            pr_act = self.state['prevact'].split('@')
            assert pr_act[0]!='inform', 'Agent called after informing!'
            act_id = dialog_config.inform_slots.index(pr_act[1])
            p_vector[self.feat_extractor.n+act_id] = 1
        p_vector = np.expand_dims(np.expand_dims(p_vector, axis=0), axis=0) # (1, 1, |Grams|+|Slots|)
        p_vector = standardize(p_vector)

        p_targets = []
        phi_targets = []
        if self.training and self.num_updates<self.pol_start:
            self._update_state(user_action['nl_sentence'], upd=self.upd, verbose=verbose)
            db_probs = self._check_db()
            H_db = tools.entropy_p(db_probs)
            H_slots = calc_entropies(self.state['inform_slots'], db_probs, self.state['database'])

            # act on policy but train on expert
            pp = np.zeros((len(dialog_config.inform_slots)+1,))
            for i,s in enumerate(dialog_config.inform_slots):
                pp[i] = H_slots[s]
            pp[-1] = H_db
            pp = np.expand_dims(np.expand_dims(pp, axis=0), axis=0) # (1, 1, |Slots|)
            _, action = self._rule_act(pp, db_probs)
            act, _, p_out, hid_out, p_db = self._prob_act(p_vector, mode='sample')
            for s in dialog_config.inform_slots:
                p_s = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum()
                p_targets.append(p_s)
                if s in self.state['dont_care']:
                    phi_targets.append(np.ones((1,)).astype('float32'))
                else:
                    phi_targets.append(np.zeros((1,)).astype('float32'))
        else:
            if self.training: act, action, p_out, hid_out, db_probs = self._prob_act(p_vector, mode='sample')
            else: act, action, p_out, hid_out, db_probs = self._prob_act(p_vector, mode='max')

        self._state_update(act, p_vector, action, user_action['reward'], p_out, hid_out, p_targets, phi_targets)
        act['posterior'] = db_probs
        return act

    def _state_update(self, act, p, action, rew, p_out, h_out, p_t, phi_t):
        '''
        每轮对话结束后，将最新的对话状态填充到列表或者更新之前存储的上一次的状态
        :param act: dict，保存了action的细节
        :param p: 每轮结束用户输入的特征和意图
        :param action: 本轮的高分action或者抽样得到的action
        :param rew: reward
        :param p_out: policy网络的RNN输出，可以作为下一轮的RNN输入
        :param h_out: BT网络RNN输出，可以作为下一次RNN的输入
        :param p_t: 手工计算出的p
        :param phi_t: 手工计算出的q
        :return: None
        '''
        if act['diaact']=='inform':
            self.state['prevact'] = 'inform@inform'
            self.state['indices'] = np.asarray(act['target'][:dialog_config.SUCCESS_MAX_RANK], dtype='int32')
        else:
            req = act['request_slots'].keys()[0]
            self.state['prevact'] = 'request@%s' %req
            self.state['num_requests'][req] += 1
        self.state['inputs'].append(p[0,0,:])
        self.state['actions'].append(action)
        self.state['rewards'].append(rew)
        self.state['hid_state'] = h_out
        self.state['pol_state'] = p_out
        self.state['ptargets'].append(p_t)
        self.state['phitargets'].append(phi_t)

    def _prob_act(self, p, mode='sample'):
        '''
        输入policy网络，policy的输出
        :param p: 输入到agent模型的input_var!
        :param mode: "sample"表示训练，"max"表示测试
        :return: 神经网络预测的aciton，policy和BT的GRU的隐状态输出和用户对于那些行感兴趣
        '''
        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        action, db_sample, db_probs, p_out, h_out, pv, phiv = self.act(p, self.state['pol_state'], self.state['hid_state'], mode=mode)
        if action==self.out_size-1:
            act['diaact'] = 'inform'
            act['target'] = [0]*self.state['database'].N
            act['target'][:dialog_config.SUCCESS_MAX_RANK] = db_sample
            act['target'][dialog_config.SUCCESS_MAX_RANK:] = list(set(range(self.state['database'].N)) - set(db_sample))
        else:
            act['diaact'] = 'request'
            s = dialog_config.inform_slots[action]
            act['request_slots'][s] = 'UNK'
        act['probs'] = pv
        act['phis'] = [phv.flatten() for phv in phiv]
        return act, action, p_out, h_out, db_probs

    def _rule_act(self, p, db_probs):
        '''
        基于规则选择action，主要是根据熵的限度和每个slot被询问的次数，优先考虑整个table的entropy，然后再考虑每个slot的entropy和询问次数
        :param p: (B, H, |Slots|)
        :param db_probs: (N,)
        :return: 选出的action，分request和inform，request需要指定具体的slot，inform不需要，还有act记录action的数据结构
        '''
        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        if p[0,0,-1] < self.tr:
            # database的熵小于一定的值，直接告知用户答案即可，不许要进行其他对话
            # agent reasonable confident, inform
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_probs)
            action = len(dialog_config.inform_slots)
        else:
            H_slots = {s:p[0,0,i] for i,s in enumerate(dialog_config.inform_slots)}
            sorted_entropies = sorted(H_slots.items(), key=operator.itemgetter(1), reverse=True)
            req = False
            for (s,h) in sorted_entropies:
                if H_slots[s]<self.frac*self.state['init_entropy'][s] or H_slots[s]<self.ts or \
                        self.state['num_requests'][s] >= self.max_req:
                    # 如果一个slot的的entropy小于一定的初始程度的一部分或小于slot的entropy下限或询问次数大于一定程度，则跳过
                    continue
                act['diaact'] = 'request'
                act['request_slots'][s] = 'UNK'
                action = dialog_config.inform_slots.index(s)
                req = True
                break
            if not req:
                # agent confident about all slots, inform
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_probs)
                action = len(dialog_config.inform_slots)
        return act, action

    def terminate_episode(self, user_action):
        '''
        当用户输入后状态变为终结时，结束本episode，将state中存储每个turn的输入信息转移到pool中，
        供update函数更新agent的参数,update函数更新参数是多个batch一起更新的，这样会更高效？
        :param user_action: 输入输入之后的对话状态
        :return: None
        '''
        assert self.state['turn'] <= self.max_turn, "More turn than MAX_TURN!!"
        total_reward = aggregate_rewards(self.state['rewards']+[user_action['reward']],self.discount)
        
        if self.state['turn']==self.max_turn:
            db_index = np.arange(dialog_config.SUCCESS_MAX_RANK).astype('int32')
            db_switch = 0
        else:
            db_index = self.state['indices']
            db_switch = 1

        inp = np.zeros((self.max_turn,self.in_size)).astype('float32')
        actmask = np.zeros((self.max_turn,self.out_size)).astype('int8')
        turnmask = np.zeros((self.max_turn,)).astype('int8')
        p_targets = [np.zeros((self.max_turn,self.slot_sizes[i])).astype('float32') \
                for i in range(len(dialog_config.inform_slots))]
        phi_targets = [np.zeros((self.max_turn,)).astype('float32') \
                for i in range(len(dialog_config.inform_slots))]
        for t in xrange(0,self.state['turn']):
            actmask[t,self.state['actions'][t]] = 1
            inp[t,:] = self.state['inputs'][t]
            turnmask[t] = 1
            if self.training and self.num_updates<self.pol_start:
                for i in range(len(dialog_config.inform_slots)):
                    p_targets[i][t,:] = self.state['ptargets'][t][i]
                    phi_targets[i][t] = self.state['phitargets'][t][i]

        self.add_to_pool(inp, turnmask, actmask, total_reward, db_index, db_switch, p_targets, phi_targets)
        self.recent_rewards.append(total_reward)
        self.recent_turns.append(self.state['turn'])
        if self.state['turn'] == self.max_turn: self.recent_successes.append(0)
        elif user_action['reward']>0: self.recent_successes.append(1)
        else: self.recent_successes.append(-1)

