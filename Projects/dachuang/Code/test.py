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
device = torch.device("cuda:0")

#传入参数
parser = argparse.ArgumentParser()
parser.add_argument("-i",default=441,type=int,help="the num of in_features")
parser.add_argument("-e",type=int,default=150,help="epoch")
parser.add_argument("-g",type=list,default=[2,3],help="g of g-gap encoding")
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

in_features = in_features * len(g)
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
print(corpus)