# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:07:02 2020

@author: xuezhiyu
"""

import numpy as np
import pandas as pd

def get_corpus(data):
    string = ""
    for i in data:
        string = string + i
    string_2 = ""
    for i in set(string):
        string_2 = string_2 + i
    return string_2

def get_corpus_g_gap(corpus):
    corpus_g_gap = []
    for i in range(len(corpus)):
        for j in range(len(corpus)):
            item = corpus[i]+corpus[j]
            corpus_g_gap.append(item)
    corpus_g_gap = dict(zip(corpus_g_gap,list(range(len(corpus_g_gap)))))
    return corpus_g_gap

def g_gap_dipeptide(data,corpus_g_gap,g_list=[2,1]):
    
    for i in range(len(g_list)):
        g = g_list[i]
        data_feature = np.array([])
        for item_index in range(len(data)):
            item = data[item_index]
            feature = [0]*len(corpus_g_gap)
            for index in range(len(item)-g):
                string = item[index] + item[index+g] #间隔为g
                # print(string)
                key = corpus_g_gap[string]
                feature[key] = feature[key] + 1
            # print(feature)
            feature = np.array(feature)
            feature = feature / sum(feature)      
            data_feature =  np.insert(data_feature,len(data_feature),values=feature,axis=0)
        data_feature = data_feature.reshape(-1,400)
        if i==0:
            data_features = data_feature
        else:
            data_features = np.concatenate((data_features,data_feature),axis=-1)
    return data_features