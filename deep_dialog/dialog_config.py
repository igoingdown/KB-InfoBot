#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
'''

all_acts = ['request', 'inform']
# TODO: 将系统内部存储的状态也转为中文
# inform_slots = ['actor','critic_rating','genre','mpaa_rating','director','release_year']
# sys_request_slots = ['actor', 'critic_rating', 'genre', 'mpaa_rating', 'director', 'release_year']
# start_dia_acts = {
#     'greeting':[],
#     'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople', 'numberofkids']
# }

inform_slots = [u'主演',u'评分',u'类别',u'评级',u'导演',u'发行年份']
sys_request_slots = [u'主演',u'评分',u'类别',u'评级',u'导演',u'发行年份']
start_dia_acts={'request':[u'片名', u'首映时间', u'影院', u'城市', u'省份', u'时间', u'类别', u'余票', u'观影人数', u'观影儿童数']}

#reward information
FAILED_DIALOG_REWARD = -1
SUCCESS_DIALOG_REWARD = 2
PER_TURN_REWARD = -0.1
SUCCESS_MAX_RANK = 5
MAX_TURN = 10

MODEL_PATH = './data/pretrained/'
