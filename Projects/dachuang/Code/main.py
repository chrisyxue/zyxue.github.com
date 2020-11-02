# coding=utf-8
"""
Created on Tue Jan 14 10:08:04 2020

@author: Zhiyu Xue
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import argparse
import os
sys.path.append(os.getcwd())
from g_gap_feature import *
from model import *
from utils import *
path = "/opt/data/private/dachuang/Data"
# path = "C:\\Users\\del\\Desktop\\大创项目\\大创项目\\data"
os.chdir(path)
# os.mkdir("result\\")
from torch.utils.data import Dataset,TensorDataset,DataLoader
import torch
import torch.optim as opt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

#传入参数
parser = argparse.ArgumentParser()
parser.add_argument("-i",default=441,type=int,help="the num of in_features")
parser.add_argument("-e",type=int,default=3000,help="epoch")
parser.add_argument("-g",type=int,default=2,help="g of g-gap encoding")
parser.add_argument("-r",type=float,default=0.8,help="the ratio of train and test")
parser.add_argument("-l",type=list,default=[300,150,100,50,10],help="layer_num_list")
parser.add_argument("-b",type=int,default=8,help="batch size")
parser.add_argument("-s",type=int,default=30,help="sample per center")
args = parser.parse_args()


in_features = args.i
Epoch = args.e
g = args.g
ratio_train_test = args.r
layer_num_list = args.l
batch_size = args.b
sample_per_center = args.s

#重要参数
# in_features = 441
# Epoch = 1000
# g = 8
# ratio_train_test = 0.8
# layer_num_list = [300,150,100,50,10]
# batch_size = 8
# sample_per_center = 30


# read data
pos = pd.read_table("anti.txt",delimiter='\n',engine="python",dtype="str")
index = list(range(0,len(pos),2))
pos = pos.iloc[index,0]
pos.index = range(len(pos))

neg = pd.read_table("nonanti.txt",delimiter='\n',engine="python",dtype="str")
index = list(range(0,len(neg),2))
neg = neg.iloc[index,0]
neg.index = range(len(neg))

# feature
corpus = get_corpus(neg)
corpus_g_gap = get_corpus_g_gap(corpus=corpus) #为一个字典，可用作速查
print(len(corpus))
print(len(corpus_g_gap))
# g-gap特征表示
g_gap_pos = torch.Tensor(g_gap_dipeptide(pos,corpus_g_gap=corpus_g_gap,g=g))
g_gap_neg = torch.Tensor(g_gap_dipeptide(neg,corpus_g_gap=corpus_g_gap,g=g))
g_gap_pos = add_y(g_gap_pos,tar=1)
g_gap_neg = add_y(g_gap_neg,tar=0)
"""
划分support和query
"""
g_gap_neg_query,g_gap_neg_support = split(g_gap_neg,len(g_gap_pos))
g_gap_neg_support = TensorDataset(g_gap_neg_support[:,:-1],g_gap_neg_support[:,-1])
"""
划分train和test
"""
g_gap_query = torch.cat((g_gap_neg_query,g_gap_pos),0)
g_gap_query_train,g_gap_query_test = split(g_gap_query,int(len(g_gap_query)*ratio_train_test)) 
#g_gap_query_train,g_gap_query_test = DataSet(g_gap_query_train),DataSet(g_gap_query_test)

train_data = TensorDataset(g_gap_query_train[:,:-1],g_gap_query_train[:,-1])
test_data = TensorDataset(g_gap_query_test[:,:-1],g_gap_query_test[:,-1])

"""
加载数据
"""
support_loader = DataLoader(
        g_gap_neg_support,
        batch_size=batch_size*sample_per_center,
        shuffle=True,
        drop_last=True
)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False,drop_last=True)

"""
模型训练与测试
"""
# 数据存储地址
name =  "structure_" + str(layer_num_list) +"_in_features_"+ str(in_features) + "_sample_center_" + str(sample_per_center) + "_g_" + str(g)
if os.path.exists("result/"+ name + "/") == False:
    os.mkdir("result/"+ name + "/")


model = Model(in_features,layer_num_list)
model.apply(weights_init_normal)
optim = opt.Adam(lr=0.01,params=model.parameters())
scheduler = StepLR(optim,step_size=150,gamma=0.1)
criterion = ContrastiveLoss()
loss_list = []
best_loss_sum = 10000 #初始值足够大
for epoch in range(Epoch):
    model.train()
    for i,data in enumerate(train_loader,0):
        inputs_support,labels_support = support_loader.__iter__().next()
        inputs,labels = data
        inputs,labels = Variable(inputs),Variable(labels)
        inputs_support,labels_support = Variable(inputs_support),Variable(labels_support)
        labels_support = labels_support[:batch_size]
        out_support = model(inputs_support)
        out_support = out_support.reshape(batch_size,sample_per_center,layer_num_list[-1])
        out_support = out_support.mean(1)
        out = model(inputs)
        labels = labels_support==labels
        labels = labels.float()
        loss = criterion(out_support,out,labels)
        scheduler.step(epoch)
        optim.zero_grad()
        loss.backward()
        optim.step()
    if epoch % 50 == 0:
        model.eval()
        loss_sum = 0
        for i,data in enumerate(test_loader,0):
            inputs_support,labels_support = support_loader.__iter__().next()
            inputs,labels = data
            inputs,labels = Variable(inputs),Variable(labels)
            inputs_support,labels_support = Variable(inputs_support),Variable(labels_support)
            labels_support = labels_support[:batch_size]
            out_support = model(inputs_support)
            out_support = out_support.reshape(batch_size,sample_per_center,layer_num_list[-1])
            out_support = out_support.mean(1)
            out = model(inputs)
            labels = labels_support==labels
            labels = labels.float()
            loss = criterion(out_support,out,labels)
            loss_sum = loss_sum + loss
            
        if best_loss_sum > loss_sum:
            save_feature(g_gap_query_train,model,"result/"+ name + "/",train=True)
            save_feature(g_gap_query_test,model,"result/"+ name + "/",train=False)
            save_model(model,"result/"+ name + "/")
            best_loss_sum = loss_sum
            print("save")
        loss_list.append(loss_sum.item())
        print("------------------")
loss_list = np.array(loss_list)
np.savetxt("result/"+ name + "/loss.txt",loss_list) #存储损失函数


# ！！！！！！！！！！！把编码后的特征存储起来



