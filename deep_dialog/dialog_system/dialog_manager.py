#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
控制对话流程
'''

import random, time
from . import MovieDict, Database
from deep_dialog import dialog_config


class DialogManager:
    def __init__(self, agent, user, db_full, db_inc, movie_kb, verbose=True):
        '''
        初始化Dialog Manager
        :param agent: agent model
        :param user: user 或者 user simulator
        :param db_full: 无missing value的database(table)
        :param db_inc: 强行加入20% missing value的database(table)
        :param movie_kb: knowledge base
        :param verbose: 决定是否打印程序运行过程中产生的冗余信息，分话痨模式和哑巴模式
        '''
        self.agent = agent
        self.user = user
        self.user_action = None
        self.database = db_full
        self.database_incomplete = db_inc
        self.verbose = verbose
        self.movie_dict = movie_kb

    def initialize_episode(self):
        '''
        初始化一次对话。用户先产生行为，包括选定目标和选定最开始的action；然后初始化agent。
        :return: 用户的action
        '''
        while True:
            self.user_action = self.user.initialize_episode()
            if self._check_user_goal()<=dialog_config.SUCCESS_MAX_RANK: break
        self.agent.initialize_episode()
        if self.verbose: self.user.print_goal()
        return self.user_action

    def next_turn(self):
        # 为了查看对话进程，可以暂时开启唠叨模式
        # self.verbose = True
        if self.verbose:
            print 'Turn', self.user_action['turn'], 'user action:', self.user_action['diaact']
            print 'inform slots:'
            for k, v in self.user_action['inform_slots'].iteritems():
                print ('{}:{}'.format(k.encode("utf8") if type(k) == unicode else k, v.encode("utf8") if v is not None and type(v) == unicode else v))
            print 'Utterance:', self.user_action['nl_sentence'].encode("utf8"), '\n'
        
        self.sys_actions = self.agent.next(self.user_action, verbose=self.verbose)
    
        self.sys_actions['turn'] = self.user_action['turn'] + 1
        if self.verbose:
            print("Turn %d sys action: %s" %(self.sys_actions['turn'], self.sys_actions['diaact']))
            for k, v in self.sys_actions['request_slots'].iteritems():
                print ('{}:{}'.format(k.encode("utf8") if type(k) == unicode else k, v.encode("utf8") if v is not None and type(v) == unicode else v))

        self.user_action, episode_over, reward = self.user.next(self.sys_actions)
        if episode_over: self.agent.terminate_episode(self.user_action)
        if episode_over and self.verbose:
            print("系统结果:")
            if 'phis' in self.sys_actions: print '\t'.join(['不关心概率:']+['%.3f'%s for s in self.sys_actions['phis']])
            if self.sys_actions['target']:
                for ii in self.sys_actions['target'][:dialog_config.SUCCESS_MAX_RANK]:
                    out = [self.database_incomplete.labels[ii]]
                    for it,slot in enumerate(self.database_incomplete.slots):
                        if 'probs' in self.sys_actions:
                            sidx = dialog_config.inform_slots.index(slot)
                            val = self.database_incomplete.tuples[ii][it]
                            idx = self.movie_dict.dict[slot].index(val) if val!='UNK' else \
                                    len(self.movie_dict.dict[slot])
                            count = self.database_incomplete.inv_counts[slot][idx]
                            out.append('%s(%.3f/%d)'%(val,self.sys_actions['probs'][sidx].flatten()[idx], \
                                    count))
                        else:
                            val = self.database_incomplete.tuples[ii][it]
                            out.append('%s'%val)
                    # 输出结果，按utf8编码
                    # print('\t'.join([o.encode('latin-1', 'replace') for o in out]))
                    print('\t'.join([v.encode("utf8") if v is not None and type(v) == unicode else v for v in out]))
        # 在进行其他操作之前，将唠叨模式关闭
        # self.verbose = False
        return (episode_over, reward, self.user_action, self.sys_actions)

    def check_db(self):
        '''

        :return:
        '''
        db_query = []
        for s in self.database.slots:
            if s in self.sys_actions['inform_slots']:
                db_query.append(self.sys_actions['inform_slots'][s])
            elif s in self.user.goal['inform_slots']:
                db_query.append(self.user.goal['inform_slots'][s])
            else:
                db_query.append(None)
        matches = self.database.lookup(db_query)
        if len(matches) > 0:
            return True
        else:
            return False

    def _check_user_goal(self):
        '''

        :return:
        '''
        db_query = []
        for s in self.database.slots:
            if s in self.user.goal['inform_slots']:
                db_query.append(self.user.goal['inform_slots'][s])
            else:
                db_query.append(None)
        matches,_ = self.database.lookup(db_query, match_unk=False)
        return len(matches)
