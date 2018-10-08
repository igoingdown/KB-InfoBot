#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, json, shutil, time, sys
import numpy as np
import importlib

from collections import Counter

agent_map = {'rule-no'  :   'nl-rule-no',
        'rl-no'         :   'simple-rl-no',
        'rule-hard'     :   'nl-rule-hard',
        'rl-hard'       :   'simple-rl-hard',
        'rule-soft'     :   'nl-rule-soft',
        'rl-soft'       :   'simple-rl-soft',
        'e2e-soft'      :   'e2e-rl-soft',
        }

EVALF = 100

parser = argparse.ArgumentParser()

parser.add_argument('--agent', dest='agent_type', type=str, default='rule-soft', 
        help='agent to use (rl-no / rl-hard / rl-soft / e2e-soft)')
parser.add_argument('--db', dest='db', type=str, default='imdb-M', 
        help='imdb-(S/M/L/XL) -- This is the KB split to use, e.g. imdb-M')
parser.add_argument('--model_name', dest='model_name', type=str, default='no_name', 
        help='model name to save')
parser.add_argument('--N', dest='N', type=int, default=500000, help='Number of simulations')
parser.add_argument('--max_turn', dest='max_turn', default=20, type=int, 
        help='maximum length of each dialog (default=20, 0=no maximum length)')
parser.add_argument('--nlg_temp', dest='nlg_temp', type=float, default=1., 
        help='Natural Language Generator softmax temperature (to control noise)')
parser.add_argument('--max_first_turn', dest='max_first_turn', type=int, default=5, 
        help='Maximum number of slots informed by user in first turn')
parser.add_argument('--err_prob', dest='err_prob', default=0.5, type=float, 
        help='the probability of the user simulator corrupting a slot value')
parser.add_argument('--dontknow_prob', dest='dontknow_prob', default=0.5, type=float, 
        help='the probability that user simulator does not know a slot value')
parser.add_argument('--sub_prob', dest='sub_prob', default=0.05, type=float, 
        help='the probability that user simulator substitutes a slot value')
parser.add_argument('--reload', dest='reload', type=int, default=0, 
        help='Reload previously saved model (0-no, 1-yes)')

args = parser.parse_args()
params = vars(args)

params['act_set'] = './data/dia_acts.txt'

# TODO: 将英文template替换为中文template
# params['template_path'] = './data/templates.p'
params['template_path'] = './data/templates_chinese.p'

params['nlg_slots_path'] = './data/nlg_slot_set.txt'
params['nlg_model_path'] = './data/pretrained/lstm_tanh_[1470015675.73]_115_120_0.657.p'

# TODO: 读取agent和data set的相关配置参数,并且加到params中，通过指定特定的agent和data set，减少需要的参数
config = importlib.import_module('settings.config_'+params['db'])
agent_params = config.agent_params
dataset_params = config.dataset_params
for k,v in dataset_params[params['db']].iteritems():
    params[k] = v
for k,v in agent_params[agent_map[params['agent_type']]].iteritems():
    params[k] = v

# print 'Dialog Parameters: '
# print json.dumps(params, indent=2)

max_turn = params['max_turn']
err_prob = params['err_prob']
dk_prob = params['dontknow_prob']
template_path = params['template_path']
agent_type = agent_map[params['agent_type']]
N = params['N']
_reload = bool(params['reload'])

datadir = './data/' + params['dataset']
# TODO: Slot set也要换成中文的
# slot_path = datadir + '/slot_set.txt'
slot_path = datadir + '/slot_set_chinese.txt'

# TODO: 将英文DB替换为中文DB
# db_full_path = datadir + '/db.txt'
# db_inc_path = datadir + '/incomplete_db_%.2f.txt' %params['unk']
# dict_path = datadir + '/dicts.json'
db_full_path = datadir + '/chinese_db.txt'
db_inc_path = datadir + '/incomplete_chinese_db_%.2f.txt' %params['unk']
dict_path = datadir + '/chinese_dicts.json'

# TODO: 把corpus也要切换到中文。
# corpus_path = './data/corpora/' + params['dataset'] + '_corpus.txt'
corpus_path = './data/corpora/' + params['dataset'] + '_chinese_corpus.txt'

from deep_dialog.dialog_system import DialogManager, MovieDict, DictReader, Database
from deep_dialog.agents import AgentSimpleRLAllAct, AgentSimpleRLAllActHardDB
from deep_dialog.agents import AgentSimpleRLAllActNoDB, AgentE2ERLAllAct
from deep_dialog.usersims import RuleSimulator, TemplateNLG, S2SNLG
from deep_dialog.objects import SlotReader
from deep_dialog import dialog_config

# 读取预先定义好的act类型request，inform，thanks，deny，并编号
act_set = DictReader()
act_set.load_dict_from_file(params['act_set'])

# 读取每个database的slot，预先定义的所有可能的槽，不同规模的dataset的槽基本相同, 以下7个是imdb-M的槽
# moviename actor critic_rating genre mpaa_rating release_year director
# 区分头实体和非头实体，头实体只有moviename，其余均为非头实体
slot_set = SlotReader(slot_path)

# 读取dict_path中已经为每个slot构建的dict
# TODO: 中文环境的dict已经生成，使用豆瓣top250的数据
# dict_path中保存的是数据集上每个槽到槽中值的list的映射，是一个2进制文件，必须通过2进制读取
movie_kb = MovieDict(dict_path)
# movie_kb实际包含了数据集上每个slot到slot value set的映射、每个slot到slot value set长度(Missing Value ID)的映射
# 以及每个slot下不同的token(不含停止词的单词)到slot value ID的映射

# 分别构造全db和按比例缺失db
db_full = Database(db_full_path, movie_kb, name=params['dataset'])
db_inc = Database(db_inc_path, movie_kb, name='incomplete%.2f_'%params['unk']+params['dataset'])

# 构造生成自然语言回复的模型
nlg = S2SNLG(template_path, params['nlg_slots_path'], params['nlg_model_path'], 
        params['nlg_temp'])

# 构造用户模拟器
user_sim = RuleSimulator(movie_kb, act_set, slot_set, None, max_turn, nlg, err_prob, db_full,
        1.-dk_prob, sub_prob=params['sub_prob'], max_first_turn=params['max_first_turn'])
# TODO: 为什么用户模拟器用db_full而agent的db用的是db_inc？
# 根据参数agent_type构建用于训练的agent和用于evaluate的agent
if agent_type == 'simple-rl-soft':
    agent = AgentSimpleRLAllAct(movie_kb, act_set, slot_set, db_inc, _reload=_reload,
            n_hid=params['nhid'],
            batch=params['batch'], ment=params['ment'], input_type=params['input'],
            pol_start=params['pol_start'],
            lr=params['lr'], upd=params['upd'], tr=params['tr'], ts=params['ts'],
            frac=params['frac'], max_req=params['max_req'], name=params['model_name'])
    agent_eval = AgentSimpleRLAllAct(movie_kb, act_set, slot_set, db_inc, train=False,
            _reload=False, n_hid=params['nhid'],
            batch=params['batch'], ment=params['ment'], input_type=params['input'],
            pol_start=params['pol_start'],
            lr=params['lr'], upd=params['upd'], tr=params['tr'], ts=params['ts'],
            frac=params['frac'], max_req=params['max_req'], name=params['model_name'])
elif agent_type == 'simple-rl-no':
    agent = AgentSimpleRLAllActNoDB(movie_kb, act_set, slot_set, db_inc, _reload=_reload,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            input_type=params['input'],
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'],
            ts=params['ts'], frac=params['frac'], max_req=params['max_req'], 
            name=params['model_name'])
    agent_eval = AgentSimpleRLAllActNoDB(movie_kb, act_set, slot_set, db_inc, train=False,
            _reload=False,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            input_type=params['input'],
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'],
            ts=params['ts'], frac=params['frac'], max_req=params['max_req'], 
            name=params['model_name'])
elif agent_type == 'simple-rl-hard':
    agent = AgentSimpleRLAllActHardDB(movie_kb, act_set, slot_set, db_inc, _reload=_reload,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            input_type=params['input'],
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'],
            ts=params['ts'], frac=params['frac'], max_req=params['max_req'], 
            name=params['model_name'])
    agent_eval = AgentSimpleRLAllActHardDB(movie_kb, act_set, slot_set, db_inc, train=False,
            _reload=False,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            input_type=params['input'],
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'],
            ts=params['ts'], frac=params['frac'], max_req=params['max_req'], 
            name=params['model_name'])
elif agent_type == 'e2e-rl-soft':
    agent = AgentE2ERLAllAct(movie_kb, act_set, slot_set, db_inc, corpus_path, _reload=_reload,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            input_type=params['input'], sl=params['sl'],
            rl=params['rl'], pol_start=params['pol_start'], lr=params['lr'], N=params['featN'],
            tr=params['tr'], ts=params['ts'], frac=params['frac'], max_req=params['max_req'],
            upd=params['upd'], name=params['model_name'])
    agent_eval = AgentE2ERLAllAct(movie_kb, act_set, slot_set, db_inc, corpus_path, train=False,
            _reload=False,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            input_type=params['input'], sl=params['sl'],
            rl=params['rl'], pol_start=params['pol_start'], lr=params['lr'], N=params['featN'],
            tr=params['tr'], ts=params['ts'], frac=params['frac'], max_req=params['max_req'],
            upd=params['upd'], name=params['model_name'])
else:
    print "Invalid agent!"
    sys.exit()

dialog_manager = DialogManager(agent, user_sim, db_full, db_inc, movie_kb, verbose=False)
dialog_manager_eval = DialogManager(agent_eval, user_sim, db_full, db_inc, movie_kb, 
        verbose=False)

def eval_agent(ite, max_perf, best=False):
    '''
    自动进行2000次对话评估agent的指标主要评估的是每次对话的平均reward
    :param ite: batch_size * 100 * N
    :param max_perf: 历史最高的测试数据reward平均值
    :param best: 区分load训练完的最好模型，还是load当前训练到某一轮的模型
    :return: max_perf，新的最高的测试数据reward平均值
    '''
    num_iter = 2000
    nn = np.sqrt(num_iter)
    # nn用于计算2000轮对话测试结果的标注差
    if best: agent_eval.load_model(dialog_config.MODEL_PATH+'best_'+agent_eval._name)
    else: agent_eval.load_model(dialog_config.MODEL_PATH+agent_eval._name)
    all_rewards = np.zeros((num_iter,))
    all_success = np.zeros((num_iter,))
    all_turns = np.zeros((num_iter,))
    for i in range(num_iter):
        current_reward = 0
        current_success = False
        utt = dialog_manager_eval.initialize_episode()
        t = 0
        while(True):
            t += 1
            episode_over, reward, utt, sact = dialog_manager_eval.next_turn()
            current_reward += reward
            if episode_over:
                if reward > 0:
                    current_success = True
                break
        all_rewards[i] = current_reward
        all_success[i] = 1 if current_success else 0
        all_turns[i] = t
    curr_perf = np.mean(all_rewards)
    print("EVAL {}: {} / {} reward {} / {} success rate {} / {} turns".format(ite,
            curr_perf, np.std(all_rewards)/nn,
            np.mean(all_success), np.std(all_success)/nn,
            np.mean(all_turns), np.std(all_turns)/nn))
    if curr_perf>max_perf and not best:
        max_perf=curr_perf
        agent_eval.save_model(dialog_config.MODEL_PATH+'best_'+agent_eval._name)
    return max_perf

print("Starting training")
mp = -10.
for i in range(N):
    if i%(EVALF*params['batch'])==0:
        mp = eval_agent(i,mp)
    utt = dialog_manager.initialize_episode()  # 在调用agent的initialization_episode函数时更新agent的模型参数
    while(True):
        episode_over, reward, utt, sact = dialog_manager.next_turn()
        if episode_over:
            break
perf = eval_agent('BEST',mp,best=True)
