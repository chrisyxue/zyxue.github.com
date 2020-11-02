# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:14:31 2020

@author: xuezhiyu
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,accuracy_score,confusion_matrix

class Model(nn.Module):
    def __init__(self,in_features,layer_num_list):
        super(Model,self).__init__()
        self.layers = nn.ModuleList()
        for hidden_feature in layer_num_list:
            layer = nn.Sequential(
                nn.Linear(in_features,hidden_feature,bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.3)
                )
            in_features = hidden_feature
            self.layers.append(layer)
            
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        out = x
        return out

    
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''
    
    classname = m.__class__.__name__
    # for every Linear layer in a model
    # m.weight.data shoud be taken from a normal distribution
    # m.bias.data should be 0
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)
        
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive

#def Prediction(output_1,output_2):
    #euclidean_distance = F.pairwise_distance(output_1, output_2)

"""
     - train_data: tensor
     - test_data: tensor
     - model: encode
"""
def Precision_Knn(train_data,test_data,model):
    x_tr,y_tr,x_te,y_te = Variable(train_data[:,:-1]),train_data[:,-1],Variable(test_data[:,:-1]),test_data[:,-1]
    x_tr,x_te = model(x_tr),model(x_te)
    classifier = KNeighborsClassifier()
    x_tr,y_tr,x_te,y_te = x_tr.detach().numpy(),y_tr.numpy(),x_te.detach().numpy(),y_te.numpy()
    print(y_tr,y_te)
    classifier.fit(x_tr,y_tr)
    y_pre = classifier.predict(x_te)
    target_name = ["0","1"]
    acc = accuracy_score(y_pre,y_te)
    return classification_report(y_te,y_pre,target_names=target_name)
    # precision = precision_score(y_pre,y_te)
    # recall = recall_score(y_te,y_pre)
    # f1 = f1_score(y_pre,y_te)
    # return precision,recall,f1

def Predict(x_te,center_pos,center_neg):
    distance_from_pos = torch.pow(x_te-center_pos,2).sum(1) #距离pos中心点的距离
    distance_from_neg = torch.pow(x_te-center_neg,2).sum(1) #距离neg中心点的距离
    y_pre = (distance_from_pos<distance_from_neg).float().numpy()
    return y_pre

def Predict(x_te,center_pos,center_neg):
    distance_from_pos = torch.pow(x_te-center_pos,2).sum(1) #距离pos中心点的距离
    distance_from_neg = torch.pow(x_te-center_neg,2).sum(1) #距离neg中心点的距离
    y_pre = (distance_from_pos<distance_from_neg).float().numpy()
    return y_pre

def Predict_gpu(x_te,center_pos,center_neg):
    distance_from_pos = torch.pow(x_te-center_pos,2).sum(1) #距离pos中心点的距离
    distance_from_neg = torch.pow(x_te-center_neg,2).sum(1) #距离neg中心点的距离
    y_pre = (distance_from_pos<distance_from_neg).float().cpu().numpy()
    return y_pre

def Precision_test(train_data,test_data,model):
    x_tr,y_tr,x_te,y_te = Variable(train_data[:,:-1]),train_data[:,-1],Variable(test_data[:,:-1]),test_data[:,-1]
    x_tr,x_te = model(x_tr),model(x_te)
    # x_tr,y_tr,x_te,y_te = x_tr.detach().numpy(),y_tr.numpy(),x_te.detach().numpy(),y_te.numpy()
    x_tr_pos = x_tr[y_tr==1]
    x_tr_neg = x_tr[y_tr==0]
    center_pos = x_tr_pos.mean(0)
    center_neg = x_tr_neg.mean(0)
    y_pre = Predict(x_te,center_pos,center_neg)
    x_tr,y_tr,x_te,y_te = x_tr.detach().numpy(),y_tr.numpy(),x_te.detach().numpy(),y_te.numpy()

    cm = confusion_matrix(y_true=y_te, y_pred=y_pre)
    tn, fp, fn, tp = cm.ravel()
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)

    acc = accuracy_score(y_te,y_pre)
    precision = precision_score(y_true=y_te,y_pred=y_pre)
    recall = recall_score(y_true=y_te,y_pred=y_pre)
    f1 = f1_score(y_true=y_te,y_pred=y_pre)
    return precision,recall,f1,sn,sp,acc
    # precision = precision_score(y_pre,y_te)
    # recall = recall_score(y_te,y_pre)
    # f1 = f1_score(y_pre,y_te)
    # return precision,recall,f1


def Precision_train(train_data,test_data,model):
    x_tr,y_tr,x_te,y_te = Variable(train_data[:,:-1]),train_data[:,-1],Variable(test_data[:,:-1]),test_data[:,-1]
    x_tr,x_te = model(x_tr),model(x_te)
    # x_tr,y_tr,x_te,y_te = x_tr.detach().numpy(),y_tr.numpy(),x_te.detach().numpy(),y_te.numpy()
    x_tr_pos = x_tr[y_tr==1]
    x_tr_neg = x_tr[y_tr==0]
    center_pos = x_tr_pos.mean(0)
    center_neg = x_tr_neg.mean(0)
    y_pre = Predict(x_tr,center_pos,center_neg)
    x_tr,y_tr,x_te,y_te = x_tr.detach().numpy(),y_tr.numpy(),x_te.detach().numpy(),y_te.numpy()
    target_name = ["0","1"]
    acc = accuracy_score(y_tr,y_pre)
    return classification_report(y_tr,y_pre,target_names=target_name),acc
    # precision = precision_score(y_pre,y_tr)
    # recall = recall_score(y_tr,y_pre)
    # f1 = f1_score(y_pre,y_tr)
    # return precision,recall,f1



def Precision_test_gpu(train_data,test_data,model,device):
    x_tr,y_tr,x_te,y_te = Variable(train_data[:,:-1]),train_data[:,-1],Variable(test_data[:,:-1]),test_data[:,-1]
    x_tr,x_te = x_tr.to(device),x_te.to(device)
    x_tr,x_te = model(x_tr),model(x_te)
    # x_tr,y_tr,x_te,y_te = x_tr.detach().numpy(),y_tr.numpy(),x_te.detach().numpy(),y_te.numpy()
    x_tr_pos = x_tr[y_tr==1]
    x_tr_neg = x_tr[y_tr==0]
    center_pos = x_tr_pos.mean(0)
    center_neg = x_tr_neg.mean(0)
    y_pre = Predict_gpu(x_te,center_pos,center_neg)
    x_tr,y_tr,x_te,y_te = x_tr.cpu().detach().numpy(),y_tr.cpu().numpy(),x_te.cpu().detach().numpy(),y_te.cpu().numpy()

    cm = confusion_matrix(y_true=y_te, y_pred=y_pre)
    tn, fp, fn, tp = cm.ravel()
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)

    acc = accuracy_score(y_te,y_pre)
    precision = precision_score(y_true=y_te,y_pred=y_pre)
    recall = recall_score(y_true=y_te,y_pred=y_pre)
    f1 = f1_score(y_true=y_te,y_pred=y_pre)
    return precision,recall,f1,sn,sp,acc


    
    
    
    
    