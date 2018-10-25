#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
'''

import lasagne
import theano
import lasagne.layers as L
import theano.tensor as T
from theano import function as F

# from theano import config
# config.compute_test_value = 'warn'

import numpy as np
import sys
import time

from deep_dialog import dialog_config

from collections import Counter, defaultdict, deque

import random
import cPickle as pkl

EPS = 1e-10

def categorical_sample(probs, mode='sample'):
    '''
    从一个多项分布中进行最大采样或者随机采样，训练模式随机采样，测试模式最大采样
    :param probs: action的概率分布
    :param mode: "sample"表示训练，"max"表示测试,
    :return: 返回抽样得到的元素的index
    '''
    if mode=='max':
        return np.argmax(probs)
    else:
        x = np.random.uniform() # 从[0,1)区间上按均匀分布随机采样出一个值
        s = probs[0]
        i = 0
        while s<x:
            i += 1
            try:
                s += probs[i]
            except IndexError:
                sys.stderr.write('Sample out of Bounds!! Probs = {} Sample = {}\n'.format(probs, x))
                return i-1
        return i

def ordered_sample(probs, N, mode='sample'):
    '''
    顺序进行N次抽样，不放回
    :param probs: 概率分布
    :param N: 抽样次数
    :param mode: "sample"表示训练，"max"表示测试
    :return: 返回抽样得到的index列表
    '''
    if mode=='max':
        return np.argsort(probs)[::-1][:N] # 将index按值降序排列后取前N个
    else:
        # 随机多项抽样，不放回
        p = np.copy(probs)
        pop = range(len(probs))
        sample = []
        for i in range(N):
            s = categorical_sample(p)
            sample.append(pop[s])
            del pop[s]
            p = np.delete(p,s)
            p = p/p.sum()
        return sample

def aggregate_rewards(rewards,discount):
    '''
    每一轮有个reward，对reward进行衰减处理
    :param rewards: 1到t轮每轮的reward
    :param discount: 衰减率
    :return: 衰变后的总的reward加和
    '''
    running_add = 0.
    for t in xrange(1,len(rewards)):
        running_add += rewards[t]*discount**(t-1)
    return running_add

class E2ERLAgent:
    #TODO: 改到embedding之后，in_size要做适当修改
    def _init_model(self, in_size, out_size, slot_sizes, db,
            n_hid=10, learning_rate_sl=0.005, learning_rate_rl=0.005, batch_size=32, ment=0.1,
            input_type='full', sl='e2e', rl='e2e'):
        '''
        初始化model，后面的工作只需要调用其中的函数或者查看其中的属性即可
        :param in_size: feature extractor 抽取出的gram的个数 GRAM_SIZE + inform slots 的个数 INFORM_SLOTS
        :param out_size: inform slots的个数 + 1，表示所有可选的action空间，前inform slots个对应request，最后一个是inform
        :param slot_sizes: 每个slot的value set的大小
        :param db: database
        :param n_hid: hidden size，RNN的隐向量的size, BF和Pol用的是一样的！
        :param learning_rate_sl: learning rate for supervised learning, 会变化，训练一定的轮数减少一半
        :param learning_rate_rl: learning rate for reinforcement learning, 0.05
        :param batch_size: batch size，单轮对话时B=1，参数更新时B=128
        :param ment: RL正则化loss的权重，在不需要正则化项的时候可以将这个超参数设为0
        :param input_type: policy network的输入是否要降维(full表示不降维，entropy表示用)
        :param sl: 在什么阶段应用SL？belief tracker，policy network 还是两个都用(e2e)
        :param rl: 在什么阶段应用RL？belief tracker，policy network 还是两个都用(e2e)
        :return: 初始化model，包括定义神经网络结构、loss、临时存储各种计算结果的列表
        '''
        self.in_size = in_size
        self.out_size = out_size
        self.slot_sizes = slot_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate_rl
        self.n_hid = n_hid
        self.r_hid = self.n_hid
        self.sl = sl
        self.rl = rl

        table = db.table
        counts = db.counts

        m_unk = [db.inv_counts[s][-1] for s in dialog_config.inform_slots]
        # m_unk实际上是每个slot的missing value的出现次数列表

        prior = [db.priors[s] for s in dialog_config.inform_slots]
        unknown = [db.unks[s] for s in dialog_config.inform_slots]
        ids = [db.ids[s] for s in dialog_config.inform_slots]

        # print("-" * 100 + '\ndb unknown:')
        # for k, v in db.unks.iteritems():
        #     print ('{}:{}'.format(k.encode("utf8") if type(k) == unicode else k,
        #                           v.encode("utf8") if v is not None and type(v) == unicode else v))
        # print("unknown rows: {}".format(unknown))
        # print("-" * 100)

        input_var, turn_mask, act_mask, reward_var = T.ftensor3('in'), T.bmatrix('tm'), T.btensor3('am'), T.fvector('r')
        T_var, N_var = T.as_tensor_variable(table), T.as_tensor_variable(counts)
        db_index_var = T.imatrix('db')
        db_index_switch = T.bvector('s')

        l_mask_in = L.InputLayer(shape=(None,None), input_var=turn_mask)
        flat_mask = T.reshape(turn_mask, (turn_mask.shape[0]*turn_mask.shape[1],1))    # (BH, 1)

        def _smooth(p):
            p_n = p+EPS
            return p_n/(p_n.sum(axis=1)[:,np.newaxis])

        def _add_unk(p,m,N):
            '''
            对p进行修正，加入UNK的分布
            :param p: (BH,|V|), p0: (1,|V|)
            :param m: num missing
            :param N: total records
            :return: 加入unk之后的p分布，(BH, |V| + 1)
            '''

            t_unk = T.as_tensor_variable(float(m)/N)  # unk的先验分布
            ps = p*(1.-t_unk)  # p的分布中，要减去unk的那部分
            return T.concatenate([ps, T.tile(t_unk, (ps.shape[0],1))], axis=1)

        def kl_divergence(p,q):
            '''
            计算第t轮由神经网络计算出来的p_j和手动计算得到的p_j(q)的KL散度
            :param p: 第t轮对话，NN Belief Tracker计算出的p_j^t, BH * |V_j|
            :param q: 第t轮对话，手动方法计算出的p_j^t， BH * |V_j|
            :return: 返回第t轮当前slot的KL散度
            '''
            p_n = _smooth(p)
            return -T.sum(q*T.log(p_n), axis=1)

        # belief tracking
        l_in = L.InputLayer(shape=(None,None,self.in_size), input_var=input_var)
        # input_var: (B, H, GRAM_SIZE + INFORM_SLOTS + 1)
        p_vars = []
        pu_vars = []
        phi_vars = []
        p_targets = []
        phi_targets = []
        hid_in_vars = []
        hid_out_vars = []
        bt_loss = T.as_tensor_variable(0.)
        kl_loss = []
        x_loss = []
        self.trackers = []
        for i,s in enumerate(dialog_config.inform_slots):
            hid_state_init = T.fmatrix('h')
            l_rnn = L.GRULayer(l_in, self.r_hid, hid_init=hid_state_init,
                    mask_input=l_mask_in,
                    grad_clipping=10.) # (B,H,D)
            # D = hidden size
            # H表示history size, 是一个MAX_TUR维，所以需要一个turn mask！
            l_b_in = L.ReshapeLayer(l_rnn, 
                    (input_var.shape[0]*input_var.shape[1], self.r_hid)) # (BH,D)

            # print("input var type: {}".format(type(input_var)))
            # print ("input var evaluation: {}".format(input_var.tag.test_value))

            hid_out = L.get_output(l_rnn)[:,-1,:]   # (B, D)
            # hid_out是RNN最终输出的output，这一点和tensorflow、pytorch是一致的！
            # print("hid out type: {}".format(type(hid_out)))

            p_targ = T.ftensor3('p_target_'+s)
            p_t = T.reshape(p_targ, 
                    (p_targ.shape[0]*p_targ.shape[1],self.slot_sizes[i]))   # (BH,|V_j|)
            phi_targ = T.fmatrix('phi_target'+s)
            phi_t = T.reshape(phi_targ, (phi_targ.shape[0]*phi_targ.shape[1], 1)) # (BH,1)

            l_b = L.DenseLayer(l_b_in, self.slot_sizes[i], 
                    nonlinearity=lasagne.nonlinearities.softmax)     # (BH,|V_j|)
            l_phi = L.DenseLayer(l_b_in, 1, 
                    nonlinearity=lasagne.nonlinearities.sigmoid)     # (BH,1)

            phi = T.clip(L.get_output(l_phi), 0.01, 0.99)
            p = L.get_output(l_b)
            # phi就是论文中的q，p就是论文中的p。

            p_u = _add_unk(p, m_unk[i], db.N)
            kl_loss.append(T.sum(flat_mask.flatten()*kl_divergence(p, p_t))/T.sum(flat_mask))
            x_loss.append(T.sum(flat_mask*lasagne.objectives.binary_crossentropy(phi,phi_t))/
                    T.sum(flat_mask))
            bt_loss += kl_loss[-1] + x_loss[-1] # (BH,)

            p_vars.append(p)
            pu_vars.append(p_u)
            phi_vars.append(phi)
            p_targets.append(p_targ)
            phi_targets.append(phi_targ)
            hid_in_vars.append(hid_state_init)
            hid_out_vars.append(hid_out)
            self.trackers.append(l_b)
            self.trackers.append(l_phi)
        self.bt_params = L.get_all_params(self.trackers)

        def check_db(pv, phi, Tb, N):
            '''
            根据p^t和q^t计算p_tao^t，即计算在第t个turn对每行感兴趣的概率并归一化
            :param pv: 第t个turn每个slot的p列表，包含Missing Value，(M,BH,|V|)
            :param phi: 第t个turn每个slot的q列表， (M,BH,1)
            :param Tb: Table
            :param N: counts
            :return: 对每个行感兴趣的列表， (BH,N)
            '''
            O = T.alloc(0.,pv[0].shape[0],Tb.shape[0]) # (BH, N)
            for i,p in enumerate(pv):
                p_dc = T.tile(phi[i], (1, Tb.shape[0]))
                O += T.log(p_dc*(1./db.table.shape[0]) + \
                        (1.-p_dc)*(p[:,Tb[:,i]]/N[np.newaxis,:,i]))
            Op = T.exp(O)#+EPS # (BH, N)
            Os = T.sum(Op, axis=1)[:,np.newaxis] # BH
            return Op/Os

        def entropy(p):
            '''
            对p进行平滑并计算entropy
            :param p: 平滑之前的参数
            :return: entropy
            '''
            p = _smooth(p)
            return -T.sum(p*T.log(p), axis=-1)

        def weighted_entropy(p,q,p0,unks,idd):
            '''
            :param p: 第t个turn的p_j^t，不包含Missing Value, (BH,|V|)
            :param q: 第t个turn对每个record感兴趣的概率列表, (BH,N)
            :param p0: 第t个turn的p_j^0，即每个slot value的先验分布，频率表示概率，不包括Missing Value, (BH,|V|)
            :param unks: 每个slot下的Missing Value所在的行号
            :param idd: 对于一个特定的slot，第i条记录的value为 v_k 时，ids[k][i] = True.(|V|, N)
            :return: p_tilde
            '''
            w = T.dot(idd,q.transpose()) # (Pi,BH)
            u = p0[np.newaxis,:]*(q[:,unks].sum(axis=1)[:,np.newaxis]) # (BH,Pi)
            p_tilde = w.transpose()+u
            return entropy(p_tilde) # (BH,)

        p_db = check_db(pu_vars, phi_vars, T_var, N_var) # (BH,N)
        
        if input_type=='entropy':
            # 计算H，对特征进行降维
            H_vars = [weighted_entropy(pv,p_db,prior[i],unknown[i],ids[i]) \
                    for i,pv in enumerate(p_vars)]
            H_db = entropy(p_db)
            phv = [ph[:,0] for ph in phi_vars]
            t_in = T.stacklists(H_vars+phv+[H_db]).transpose() # (BH,2M+1)
            t_in_resh = T.reshape(t_in, (turn_mask.shape[0], turn_mask.shape[1], t_in.shape[1])) # (B,H,2M+1)
            l_in_pol = L.InputLayer(shape=(None,None,2*len(dialog_config.inform_slots)+1), input_var=t_in_resh)
        else:
            # 不对特征进行降维，特征维度非常高，参数量极大！
            in_reshaped = T.reshape(input_var, (input_var.shape[0]*input_var.shape[1], input_var.shape[2]))
            prev_act = in_reshaped[:,-len(dialog_config.inform_slots):]
            t_in = T.concatenate(pu_vars+phi_vars+[p_db,prev_act], 
                    axis=1) # (BH,D-sum+A)
            t_in_resh = T.reshape(t_in, (turn_mask.shape[0], turn_mask.shape[1], t_in.shape[1])) # (B,H,D-sum)
            l_in_pol = L.InputLayer(shape=(None,None,sum(self.slot_sizes)+ \
                    3*len(dialog_config.inform_slots)+ \
                    table.shape[0]), input_var=t_in_resh)

        pol_in = T.fmatrix('pol-h')
        l_pol_rnn = L.GRULayer(l_in_pol, n_hid, hid_init=pol_in, mask_input=l_mask_in, grad_clipping=10.) # (B,H,D)
        pol_out = L.get_output(l_pol_rnn)[:,-1,:] # (B,D)
        l_den_in = L.ReshapeLayer(l_pol_rnn, (turn_mask.shape[0]*turn_mask.shape[1], n_hid)) # (BH,D)
        l_out = L.DenseLayer(l_den_in, self.out_size, nonlinearity=lasagne.nonlinearities.softmax) # (BH,A)
        # A 表示Action sample size！

        self.network = l_out
        self.pol_params = L.get_all_params(self.network)
        self.params = self.bt_params + self.pol_params

        # db loss
        p_db_reshaped = T.reshape(p_db, (turn_mask.shape[0],turn_mask.shape[1],table.shape[0])) # (B,H,N)
        p_db_final = p_db_reshaped[:,-1,:] # (B,N)
        p_db_final = _smooth(p_db_final)
        ix = T.tile(T.arange(p_db_final.shape[0]),(db_index_var.shape[1],1)).transpose()   # (B,db_index_var.shape[1])
        sample_probs = p_db_final[ix,db_index_var] # (B,K)
        # db_loss对应论文中的miu，是强化学习的一部分，详细步骤可以参考论文的附录C部分
        # foo = F([],[db_index_var, db_index_switch])
        # print("-" * 200 + "\nix, db_index_var and sample probs: {}\n".format(foo()) + "-" * 200)

        # SUCCESS_MAX_RANK表示允许正确答案最多排到结果列表的第几位，默认值为5
        if dialog_config.SUCCESS_MAX_RANK==1:
            log_db_probs = T.log(sample_probs).sum(axis=1)
        else:
            cum_probs,_ = theano.scan(fn=lambda x, prev: x+prev,
                    outputs_info=T.zeros_like(sample_probs[:,0]),
                    sequences=sample_probs[:,:-1].transpose())
            cum_probs = T.clip(cum_probs.transpose(), 0., 1.-1e-5) # (B,K-1)
            log_db_probs = T.log(sample_probs).sum(axis=1) - T.log(1.-cum_probs).sum(axis=1) # (B,)
        log_db_probs = log_db_probs * db_index_switch

        # rl
        probs = L.get_output(self.network) # (BH,A)
        probs = _smooth(probs)
        out_probs = T.reshape(probs, (turn_mask.shape[0],turn_mask.shape[1],self.out_size)) # (B,H,A)
        log_probs = T.log(out_probs)
        act_probs = (log_probs*act_mask).sum(axis=2) # (B,H)
        ep_probs = (act_probs*turn_mask).sum(axis=1) # (B,)
        H_probs = -T.sum(T.sum(out_probs*log_probs,axis=2),axis=1) # Entropy, (B,)
        self.act_loss = -T.mean(ep_probs*reward_var)
        self.db_loss = -T.mean(log_db_probs*reward_var)
        # reg_loss是正则化项，但是论文里并没提到，这是因为ment默认值为0
        self.reg_loss = -T.mean(ment*H_probs)
        self.loss = self.act_loss + self.db_loss + self.reg_loss

        self.inps = [input_var, turn_mask, act_mask, reward_var, db_index_var, db_index_switch, pol_in] + hid_in_vars   # inputs
        self.obj_fn = theano.function(self.inps, self.loss, on_unused_input='warn')
        self.act_fn = theano.function([input_var,turn_mask,pol_in]+hid_in_vars, [out_probs,p_db,pol_out]+pu_vars+phi_vars+hid_out_vars, on_unused_input='warn')
        self.debug_fn = theano.function(self.inps, [probs, p_db, self.loss], on_unused_input='warn')
        self._rl_train_fn(self.learning_rate)

        ## sl
        sl_loss = 0. + bt_loss - T.mean(ep_probs) 

        if self.sl=='e2e':
            sl_updates = lasagne.updates.rmsprop(sl_loss, self.params,
                    learning_rate=learning_rate_sl, epsilon=1e-4)
            sl_updates_with_mom = lasagne.updates.apply_momentum(sl_updates)
        elif self.sl=='bel':
            sl_updates = lasagne.updates.rmsprop(sl_loss, self.bt_params,
                    learning_rate=learning_rate_sl, epsilon=1e-4)
            sl_updates_with_mom = lasagne.updates.apply_momentum(sl_updates)
        else:
            sl_updates = lasagne.updates.rmsprop(sl_loss, self.pol_params,
                    learning_rate=learning_rate_sl, epsilon=1e-4)
            sl_updates_with_mom = lasagne.updates.apply_momentum(sl_updates)

        sl_inps = [input_var, turn_mask, act_mask, pol_in] + p_targets + phi_targets + hid_in_vars
        self.sl_train_fn = theano.function(sl_inps, [sl_loss]+kl_loss+x_loss, updates=sl_updates,
                on_unused_input='warn')
        self.sl_obj_fn = theano.function(sl_inps, sl_loss, on_unused_input='warn')

    def _rl_train_fn(self, lr):
        '''
        按照rl的方式更新参数
        :param lr: learning rate
        :return: None
        '''
        if self.rl=='e2e':
            updates = lasagne.updates.rmsprop(self.loss, self.params, learning_rate=lr, epsilon=1e-4)
            updates_with_mom = lasagne.updates.apply_momentum(updates)
        elif self.rl=='bel':
            updates = lasagne.updates.rmsprop(self.loss, self.bt_params, learning_rate=lr,
                    epsilon=1e-4)
            updates_with_mom = lasagne.updates.apply_momentum(updates)
        else:
            updates = lasagne.updates.rmsprop(self.loss, self.pol_params, learning_rate=lr,
                    epsilon=1e-4)
            updates_with_mom = lasagne.updates.apply_momentum(updates)
        self.train_fn = theano.function(self.inps, [self.act_loss,self.db_loss,self.reg_loss],
                updates=updates)

    def train(self, inp, tur, act, rew, db, dbs, pin, hin):
        '''
        训练模型
        :param inp: input
        :param tur: turn
        :param act: action
        :param rew: reward
        :param db: episode的每个turn后agent给用户推荐的答案
        :param dbs: 上述的答案是否有效，作为开关
        :param pin: current policy RNN hidden state
        :param hin: current BF RNN hidden state
        :return: RL的三部分的loss
        '''
        return self.train_fn(inp, tur, act, rew, db, dbs, pin, *hin)

    def sl_train(self, inp, tur, act, pin, ptargets, phitargets, hin):
        '''
        训练初期使用SL的方式训练模型，监督信号是手工计算的p和q
        :param inp: input
        :param tur: turn
        :param act: action
        :param pin: current policy RNN hidden state
        :param ptargets: 手工计算的p
        :param phitargets: 手工计算的q
        :param hin: current BF RNN hidden state
        :return: 返回计算得到的loss，并更新参数
        '''
        return self.sl_train_fn(inp, tur, act, pin, *ptargets+phitargets+hin)

    def evaluate(self, inp, tur, act, rew, db, dbs, pin, hin):
        '''
        通过计算模型的loss评测模型
        :param inp: input，原始特征
        :param tur: turn
        :param act: action
        :param rew: reward
        :param db:episode的每个turn后agent给用户推荐的答案
        :param dbs: 上述的答案是否有效，作为开关
        :param pin: current policy RNN hidden state
        :param hin: current BF RNN hidden state
        :return: 返回模型的loss，不更新参数
        '''
        return self.obj_fn(inp, tur, act, rew, db, dbs, pin, *hin)

    def act(self, inp, pin, hin, mode='sample'):
        '''
        将输入数据到模型中，为训练和测试分别抽样得到后续的action并更新的p，q以及RNN的隐状态，这些都是对话状态！
        :param inp: input，原始特征，(B, H, |Grams + slots|)
        :param pin: current policy RNN hidden state
        :param hin: current BF RNN hidden state
        :param mode: 训练时设为"sample"，测试时设为"max"
        :return: 抽样得到的action和db的记录，db的概率（用户对于每一行感兴趣的概率）,policy 和 Belief部分RNN的输出状态、每个slot的p和q
        '''
        tur = np.ones((inp.shape[0],inp.shape[1])).astype('int8') # (B,H)
        outs = self.act_fn(inp, tur, pin, *hin)
        act_p, db_p, p_out = outs[0], outs[1], outs[2]
        n_slots = len(dialog_config.inform_slots)
        pv = outs[3:3+n_slots]
        phiv = outs[3+n_slots:3+2*n_slots]
        h_out = outs[3+2*n_slots:]
        action = categorical_sample(act_p.flatten(), mode=mode)
        if action==self.out_size-1:
            db_sample = ordered_sample(db_p.flatten(), dialog_config.SUCCESS_MAX_RANK, mode=mode)
        else:
            db_sample = []
        return action, db_sample, db_p.flatten(), p_out, h_out, pv, phiv

    def sl_evaluate(self, inp, tur, act, pin, ptargets, phitargets, hin):
        return self.sl_obj_fn(inp, tur, act, pin, *ptargets+phitargets+hin)

    def anneal_lr(self):
        '''
        训练次数每到达一定的值，就讲lr减小一半，这种方式相当不智能！不如使用Adam的优化器！
        :return:
        '''
        self.learning_rate /= 2.
        self._rl_train_fn(self.learning_rate)

    def _debug(self, inp, tur, act, rew, beliefs):
        print 'Input = {}, Action = {}, Reward = {}'.format(inp, act, rew)
        out = self.debug_fn(inp, tur, act, rew, *beliefs)
        for item in out:
            print item

    def _init_experience_pool(self, pool):
        '''
        初始化各种pool为deque，最大长度为batch_size，用于记录最近batch_size个episode。
        :param pool: batch size, 128 by default, 下面作为deque的最大长度
        :return: None
        '''
        self.input_pool = deque([], pool)
        self.actmask_pool = deque([], pool)
        self.reward_pool = deque([], pool)
        self.db_pool = deque([], pool)
        self.dbswitch_pool = deque([], pool)
        self.turnmask_pool = deque([], pool)
        self.ptarget_pool = deque([], pool)
        self.phitarget_pool = deque([], pool)

    def add_to_pool(self, inp, turn, act, rew, db, dbs, ptargets, phitargets):
        '''

        :param inp: 一个episode的用户输入，N-Gram自然语言特征
        :param turn: 一个episode的轮次列表
        :param act: 一个episode中agent的action列表
        :param rew: 一个episode中agent的reward列表
        :param db: 当前episode的每个turn的输出的答案
        :param dbs: 当前episode的每个turn是否输出了答案，如果没有为0表示该turn关闭，如果为1表示开启
        :param ptargets: 一个episode中手工计算的p列表
        :param phitargets: 一个episode中手工计算的q列表
        :return: None，原地修改各个pool
        '''
        self.input_pool.append(inp)
        self.actmask_pool.append(act)
        self.reward_pool.append(rew)
        self.db_pool.append(db)
        self.dbswitch_pool.append(dbs)
        self.turnmask_pool.append(turn)
        self.ptarget_pool.append(ptargets)
        self.phitarget_pool.append(phitargets)

    def _get_minibatch(self, N):
        '''
        对话过程中是episode一个接一个产生的，此时的B=1。模型更新时则是采用最新的最多batch_size个episode用于更新参数，此时B=128。
        input_pool是一个capacity为batch_size的deque，无论deque是否满了，本函数都将input_pool进行shuffle，
        将pool中存储的数据用于更新模型参数
        :param N: batch_size
        :return: 重新抽样得到的数据
        '''
        n = min(N, len(self.input_pool))
        index = random.sample(range(len(self.input_pool)), n)
        i = [self.input_pool[ii] for ii in index]
        a = [self.actmask_pool[ii] for ii in index]
        r = [self.reward_pool[ii] for ii in index]
        d = [self.db_pool[ii] for ii in index]
        ds = [self.dbswitch_pool[ii] for ii in index]
        t = [self.turnmask_pool[ii] for ii in index]
        p = [self.ptarget_pool[ii] for ii in index]
        pp = [np.asarray([row[ii] for row in p], dtype='float32') for ii in range(len(p[0]))]
        ph = [self.phitarget_pool[ii] for ii in index]
        pph = [np.asarray([row[ii] for row in ph], dtype='float32') for ii in range(len(ph[0]))]
        return np.asarray(i, dtype='float32'), np.asarray(t, dtype='int8'), np.asarray(a, dtype='int8'), np.asarray(r, dtype='float32'), np.asarray(d, dtype='int32'), np.asarray(ds, dtype='int8'), pp, pph

    def update(self, verbose=False, regime='RL'):
        '''
        训练模型，并更新参数
        :param verbose: 是否开启唠叨模式以便debug
        :param regime: 用SL的方式训练还是使用RL的方式训练
        :return: loss，RL模式返回action(policy) loss， database sample(belief tracker) loss 和 正则化项loss， SL模式返回
        '''
        i, t, a, r, d, ds, p, ph = self._get_minibatch(self.batch_size)
        hi = [np.zeros((1,self.r_hid)).astype('float32') \
                for s in dialog_config.inform_slots]
        pi = np.zeros((1,self.n_hid)).astype('float32')

        if verbose: print i, t, a, r, d, ds, p, ph, hi
        if regime=='RL':
            r -= np.mean(r)
            al,dl,rl = self.train(i,t,a,r,d,ds,pi,hi)
            g = al+dl+rl
        else:
            g = self.sl_train(i,t,a,pi,p,ph,hi)
        return g

    def eval_objective(self, N):
        try:
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r, self.eval_b)
        except AttributeError:
            self.eval_i, self.eval_t, self.eval_a, self.eval_r, self.eval_b = self._get_minibatch(N)
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r, self.eval_b)
        return obj

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pkl.load(f)
            L.set_all_param_values(self.network, data)
            for item in self.trackers:
                data = pkl.load(f)
                L.set_all_param_values(item, data)

    def save_model(self, save_path):
        with open(save_path, 'w') as f:
            data = L.get_all_param_values(self.network)
            pkl.dump(data, f)
            for item in self.trackers:
                data = L.get_all_param_values(item)
                pkl.dump(data, f)
