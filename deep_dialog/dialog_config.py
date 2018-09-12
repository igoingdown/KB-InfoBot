#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
'''

all_acts = ['request', 'inform']
inform_slots = ['actor','critic_rating','genre','mpaa_rating','director','release_year']
chinese_inform_slots = ['演员','IMDB评分','类别','MPAA评级','导演','发行时间']

sys_request_slots = ['actor', 'critic_rating', 'genre', 'mpaa_rating', 'director', 'release_year']
chinese_sys_request_slots = ['演员','IMDB评分','类别','MPAA评级','导演','发行时间']

start_dia_acts = {
    #'greeting':[],
    'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople', 'numberofkids']
}
chinese_start_dia_acts={'request':['影片名称', '首映时间', '影院', '城市', '省份', '时间', '类别', '余票', '观影人数', '观影儿童数']
}

#reward information
FAILED_DIALOG_REWARD = -1
SUCCESS_DIALOG_REWARD = 2
PER_TURN_REWARD = -0.1
SUCCESS_MAX_RANK = 5
MAX_TURN = 10

MODEL_PATH = './data/pretrained/'
