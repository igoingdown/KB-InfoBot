#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
'''

import io
import nltk
import numpy as np
import cPickle as pkl
import os.path
import string
import torch
import torchwordemb

from deep_dialog.tools import to_tokens

def test_torch_load_word2vec(embedding_path):
    '''
    测试torch在load word2vec模型时是不是好用，事实证明很好用，其实跟word2vec内嵌的API效果相同。
    :param config: 配置项超参数
    :return: None
    '''
    vocab, embedding  = torchwordemb.load_word2vec_text(embedding_path)
    print(embedding[vocab[u'中国']])


class FeatureExtractor:
    def __init__(self, corpus_path, db_path, embedding_path='../data/embedding_model_t2s/vector_t2s', N=1):
        self.N = N
        self.embedding_path = embedding_path
        self.embedding_vocab_t2n, self.embedding_vectors = torchwordemb.load_word2vec_text(self.embedding_path)
        self.embedding_size = self.embedding_vectors.size()[-1]
        save_path = db_path.rsplit('/',1)[0] + '/fdict_%d.p'%N
        pure_grams_save_path = db_path.rsplit('/',1)[0] + '/fdict_%d.p'%2
        print(save_path, pure_grams_save_path)
        if os.path.isfile(save_path):
            # load pre-dumped grams and N from file
            print("dict exist")
            f = open(save_path, 'rb')
            self.grams = pkl.load(f)
            self.n = pkl.load(f)
            f.close()
        else:
            print("build dicts")
            # if there doesn't exist any pre-dumped file, generate grams from corpus or database text file.
            self.grams = {}
            self.n = 0
            if corpus_path is not None: self._build_vocab_from_corpus(corpus_path)
            # print("vocab size after corpus building: {}".format(self.n))
            if db_path is not None: self._build_vocab_from_db(db_path)
            # print("vocab size after db building:{}".format(self.n))
            with open(save_path, 'wb') as f:
                pkl.dump(self.grams, f)
                pkl.dump(self.n, f)
        print("initialize vocab size:{}".format(self.n))

    def _build_vocab_from_db(self, corpus):
        '''
        根据database构造全局vocabulary存到grams中，N-Gram实际上包括了全部[1,N]-Grams
        :param corpus: database文本文件path
        :return: None
        '''

        # 中文中就不需要N-Gram了！
        with open(corpus, 'r') as f:
            for line in f:
                elements = line.strip().split('\t')[1:]
                for ele in elements:
                    # if '·' in ele:
                    #     tokens = ele.split('·')
                    # else:
                    tokens = to_tokens(ele)
                    for ngram in tokens:
                        if ngram.strip() != "" and ngram not in self.grams:
                            print(ngram.encode("utf8") if ngram is not None and type(ngram) == unicode else ngram)
                            self.grams[ngram] = self.n
                            self.n += 1

        # try:
        #     f = io.open(corpus, 'r')
        #     for line in f:
        #         elements = line.rstrip().split('\t')[1:]
        #         for ele in elements:
        #             tokens = to_tokens(ele)
        #             for i in range(len(tokens)):
        #                 for t in range(self.N):
        #                     if i-t<0: continue
        #                     ngram = '_'.join(tokens[i-t:i+1])
        #                     if ngram not in self.grams:
        #                         self.grams[ngram] = self.n
        #                         self.n += 1
        #     f.close()
        # except UnicodeDecodeError:
        #     f = open(corpus, 'r')
        #     for line in f:
        #         elements = line.rstrip().split('\t')[1:]
        #         for ele in elements:
        #             tokens = to_tokens(ele)
        #             for i in range(len(tokens)):
        #                 for t in range(self.N):
        #                     if i-t<0: continue
        #                     ngram = '_'.join(tokens[i-t:i+1])
        #                     if ngram not in self.grams:
        #                         self.grams[ngram] = self.n
        #                         self.n += 1
        #     f.close()

    def _build_vocab_from_corpus(self, corpus):
        '''
        根据对话文本构造全局vocabulary存到grams中，N-Gram实际上包括了全部[1,N]-Grams
        :param corpus: 对话文本文件path
        :return: None
        '''

        #转到中文之后，N-Gram不再必要了
        if not os.path.isfile(corpus): return
        with open(corpus, 'r') as f:
            for line in f:
                for ngram in to_tokens(line.strip()):
                    if ngram.strip() != "" and ngram not in self.grams:
                        self.grams[ngram] = self.n
                        print(ngram.encode("utf8") if ngram is not None and type(ngram) == unicode else ngram)
                        self.n += 1

        # try:
        #     f = io.open(corpus, 'r')
        #     for line in f:
        #         tokens = to_tokens(line.rstrip())
        #         for i in range(len(tokens)):
        #             for t in range(self.N):
        #                 if i-t<0: continue
        #                 ngram = '_'.join(tokens[i-t:i+1])
        #                 if ngram not in self.grams:
        #                     self.grams[ngram] = self.n
        #                     self.n += 1
        #     f.close()
        # except UnicodeDecodeError:
        #     f = open(corpus, 'r')
        #     for line in f:
        #         tokens = to_tokens(line.rstrip())
        #         for i in range(len(tokens)):
        #             for t in range(self.N):
        #                 if i-t<0: continue
        #                 ngram = '_'.join(tokens[i-t:i+1])
        #                 if ngram not in self.grams:
        #                     self.grams[ngram] = self.n
        #                     self.n += 1
        #     f.close()

    def featurize(self, text):
        '''
        基于N-Gram的方式构造文本text中的特征
        :param text: 自然语言文本
        :return: 长度为|Grams|的向量，向量中的每个值表示该Gram的数量
        '''
        vec = np.zeros((len(self.grams),)).astype('float32')

        embeddings = []
        UNK_EMBEDDING = self.embedding_vectors.mean(0).squeeze()
        BAK_EMBEDDING = torch.zeros(UNK_EMBEDDING.size())
        # 转到中文之后，N-Gram不再必要了
        for ngram in to_tokens(text):
            if ngram in self.grams:
                vec[self.grams[ngram]] += 1.
            if ngram in self.embedding_vocab_t2n:
                embeddings.append(self.embedding_vectors[self.embedding_vocab_t2n[ngram]])
            else:
                embeddings.append(UNK_EMBEDDING)
        average_embedding = torch.cat([x.view(1, x.size()) for x in embeddings], 0).mean(0).squeeze()
        # tokens = to_tokens(text)
        # for i in range(len(tokens)):
        #     for t in range(self.N):
        #         if i-t<0: continue
        #         ngram = '_'.join(tokens[i-t:i+1])
        #         if ngram in self.grams:
        #             vec[self.grams[ngram]] += 1.
        return vec

if __name__=='__main__':
    F = FeatureExtractor('../data/corpora/selected_medium_corpus.txt','../data/selected_medium/db.txt')
    print '\n'.join(F.grams.keys())
    print F.featurize('Please search for the movie with Matthew Saville as director')
    print F.featurize('I would like to see the movie with drama as genre')
