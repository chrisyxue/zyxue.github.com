# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:06:29 2020

@author: xuezhiyu
"""
import numpy as np
import torch
import random
random.seed(1)
# 把数据分成两部分
def split(data,threshold):
    index = list(range(len(data)))
    random.shuffle(index)
    index_1 = index[0:threshold]
    index_2 = index[threshold:]
    data_1 = data[index_1,:]
    data_2 = data[index_2,:]
    return data_1,data_2

def add_y(data,tar=1):
    y = torch.Tensor([tar]*len(data)).reshape(-1,1)
    out = torch.cat((data,y),1)
    return out

def save_feature(data,model,path,train=True):
    x = data[:,:-1]
    y = data[:,-1]
    feature = model(x)
    feature = torch.cat((feature,y.reshape(-1,1)),1)
    feature = feature.detach().numpy()
    if train == True:
        file_path = path + "train_feature.txt"
    else:
        file_path = path + "test_feature.txt"
    # print(file_path)
    np.savetxt(file_path,feature)
    
def save_model(model,path):
    file_path = path + "model.pkl"
    torch.save(model.state_dict(),file_path)
    

    
    
