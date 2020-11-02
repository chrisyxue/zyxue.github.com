# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:38:10 2020

@author: xuezhiyu
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
import random

#传入参数
parser = argparse.ArgumentParser()
parser.add_argument("-i",default=441,type=int,help="the num of in_features")
parser.add_argument("-e",type=int,default=200,help="epoch")
parser.add_argument("-g",type=int,default=2,help="g of g-gap encoding")
parser.add_argument("-r",type=float,default=0.8,help="the ratio of train and test")
parser.add_argument("-l",type=list,default=[300,100,50,10],help="layer_num_list")
parser.add_argument("-b",type=int,default=32,help="batch size")
parser.add_argument("-lr",type=int,default=0.005,help="learning rate")
parser.add_argument("-m",type=float,default=2.0,help="margin of the loss")
args = parser.parse_args()


in_features = args.i
Epoch = args.e
g = args.g
ratio_train_test = args.r
layer_num_list = args.l
batch_size = args.b
LR = args.lr
margin = args.m

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
# g_gap_neg_query,g_gap_neg_support = split(g_gap_neg,len(g_gap_pos))
# g_gap_neg_support = TensorDataset(g_gap_neg_support[:,:-1],g_gap_neg_support[:,-1])

"""
划分train和test
"""
g_gap_all = torch.cat((g_gap_neg,g_gap_pos),0)
g_gap_train,g_gap_test = split(g_gap_all,int(len(g_gap_all)*ratio_train_test)) 
#g_gap_query_train,g_gap_query_test = DataSet(g_gap_query_train),DataSet(g_gap_query_test)

"""
封装数据
"""
# train_data = TensorDataset(g_gap_query_train[:,:-1],g_gap_query_train[:,-1])
# test_data = TensorDataset(g_gap_query_test[:,:-1],g_gap_query_test[:,-1])
    
class MyDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.x = data[:,:-1]
        self.y = data[:,-1]
        self.num = len(data) #有多少个样本
    def __getitem__(self,index):
        x_1,y_1 = self.x[index],self.y[index]
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            y_2 = y_1 # 样本属于同一类别
            samples = self.x[self.y==y_2]
            index_2 = random.randint(0,len(samples)-1)
            x_2 = samples[index_2]
        else:
            y_2 = ((y_1-1)<0).float() # 取反
            samples = self.x[self.y==y_2]
            index_2 = random.randint(0,len(samples)-1)
            x_2 = samples[index_2]
        y = (y_2==y_1).float()
        return x_1,x_2,y
    def __len__(self):
        return len(self.data)


train_data = MyDataset(g_gap_train)
test_data = MyDataset(g_gap_test)


"""
加载数据
"""
# support_loader = DataLoader(
        # g_gap_neg_support,
        # batch_size=batch_size*sample_per_center,
        # shuffle=True,
        # drop_last=True
# )

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
test_loader = DataLoader(test_data,batch_size=1,shuffle=False)

"""
模型训练与测试
"""
# 数据存储地址
name =  "structure_" + str(layer_num_list) +"_in_features_"+ str(in_features) + "_g_" + str(g) + "_lr_" + str(LR) + "_m_" + str(margin) + "_train_test_" +str(ratio_train_test)
if os.path.exists("result_euclidean/"+ name + "/") == False:
    os.mkdir("result_euclidean/"+ name + "/")


model = Model(in_features,layer_num_list)
model.apply(weights_init_normal)
optim = opt.Adam(lr=LR,params=model.parameters())
scheduler = StepLR(optim,step_size=50,gamma=0.1)
criterion = ContrastiveLoss(margin=margin)
loss_list = []
acc_list = []
precision_list = []
recall_list = []
f1_list = []
sp_list = []
sn_list = []

best_acc = 0 #初始值足够大
for epoch in range(Epoch):
    model.train()
    for i,data in enumerate(train_loader,0):
        x_1,x_2,labels = data
        x_1,x_2,labels = Variable(x_1),Variable(x_2),Variable(labels)
        out_1,out_2 = model(x_1),model(x_2)
        loss = criterion(out_1,out_2,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step(epoch)
        # print(loss)
    if epoch % 1 == 0:
        model.eval()
        loss_sum = 0
        for i,data in enumerate(test_loader,0):
            x_1,x_2,labels = data
            x_1,x_2,labels = Variable(x_1),Variable(x_2),Variable(labels)
            out_1,out_2 = model(x_1),model(x_2)
            loss = criterion(out_1,out_2,labels)
            loss_sum = loss_sum + loss
        precision,recall,f1,sn,sp,acc = Precision_test(g_gap_train,g_gap_test,model)
        if best_acc < acc:
            save_feature(g_gap_train,model,"result_euclidean/"+ name + "/",train=True)
            save_feature(g_gap_test,model,"result_euclidean/"+ name + "/",train=False)
            save_model(model,"result_euclidean/"+ name + "/")
            best_acc = acc
            print("save")
        print(precision,recall,f1,sn,sp,acc)
        
        loss_mean = loss_sum / len(g_gap_test)
        loss_list.append(loss_mean.item())
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        sn_list.append(sn)
        sp_list.append(sp)
        # print("------------------")
        print(loss_mean)
        # report,acc = Precision_train(g_gap_train,g_gap_test,model)
        # print(acc)
        # print(report)
        print("------------------")

loss_list = np.array(loss_list)
acc_list = np.array(acc_list)
precision_list = np.array(precision_list)
recall_list = np.array(recall_list)
f1_list = np.array(f1_list)
sp_list = np.array(sp_list)
sn_list = np.array(sn_list)

np.savetxt("result_euclidean/"+ name + "/loss.txt",loss_list) #存储损失函数
np.savetxt("result_euclidean/"+ name + "/acc.txt",acc_list)
np.savetxt("result_euclidean/"+ name + "/precision.txt",precision_list)
np.savetxt("result_euclidean/"+ name + "/recall.txt",recall_list)
np.savetxt("result_euclidean/"+ name + "/f1.txt",f1_list)
np.savetxt("result_euclidean/"+ name + "/sp.txt",sp_list)
np.savetxt("result_euclidean/"+ name + "/sn.txt",sn_list)
