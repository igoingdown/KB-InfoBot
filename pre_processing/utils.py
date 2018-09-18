#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
生成中文的movie dict，并dump下来
'''

import pickle, os
import random

def generate_dict_from_db(db_path):
    d = {}
    d[u'片名'] = set()
    d[u'主演'] = set()
    d[u'评分'] = set()
    d[u'类别'] = set()
    d[u'评级'] = set()
    d[u'发行年份'] = set()
    d[u'导演'] = set()

    line_num = 0
    with open(db_path, "r") as f:
        for line in f:
            if line_num != 0:
                line = line.decode("utf-8")
                movie_name,actor,critic_rating,genre,mpaa_rating,release_year,director = line.split("\t")
                if movie_name != u"UNK": d[u'片名'].add(movie_name.strip())
                if actor != u"UNK": d[u'主演'].add(actor.strip())
                if critic_rating != u"UNK": d[u'评分'].add(critic_rating.strip())
                if genre != u"UNK": d[u'类别'].add(genre.strip())
                if mpaa_rating != u"UNK": d[u'评级'].add(mpaa_rating.strip())
                if release_year != u"UNK": d[u'发行年份'].add(release_year.strip())
                if director != u"UNK": d[u'导演'].add(director.strip())
            line_num += 1
    for key in d.keys():
        d[key] = list(d[key])
    return d

def dump_file(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_obj_from_file(path):
    if (os.path.exists(path)):
        with open(path, "rb") as f:
            return pickle.load(f)

def generate_incomplete_database(db_path, incomplete_db_path):
    with open(incomplete_db_path, "wb") as w_f:
        with open (db_path, "r") as f:
            line_num = 0
            for line in f:
                line = line.decode("utf-8")
                if line_num != 0:
                    items = [x.strip() for x in line.split("\t")]
                    sampled_index = random.sample(range(7), 1)[0]
                    while sampled_index == 0 or items[sampled_index] == u'UNK':
                        sampled_index = random.sample(range(7), 1)[0]
                    items[sampled_index] = u'UNK'
                    str = u'\t'.join(items) + u'\n'
                    w_f.write(str.encode("utf-8"))
                else:
                    w_f.write(line.encode("utf-8"))
                line_num += 1

        pass

if __name__ == '__main__':
    d = generate_dict_from_db("data/chinese_db.txt")
    dump_file("data/chinese_dicts.json", d)
    p = load_obj_from_file("data/chinese_dicts.json")
    generate_incomplete_database("data/chinese_db.txt", "data/incomplete_chinese_db_0.20.txt")
    for key, v in p.iteritems():
        print key
        for item in v:
            print item
    print p

