#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:55:25 2023

@author: sergio
"""
"""
pytorch implementaiton of four SAR assessing metrics.

Variable Defintion:

all the quantity are in intensity format
all the quantity are considered as tensor of four dimensions [batch_size, 1, rows, cols]

inputs: SAR image used as input of the DL method
outputs: noise-free estimate provided as output of the DL method
ref:	(only for supervised methods) noise-free reference

"""
import torch
from torch.autograd import Variable
import sys
eps = sys.float_info.epsilon
import numpy as np
import math

#no-reference version (for unsupervised and supervised methods)
def enl_loss(outputs):
    mu = torch.mean(outputs**2,dim=(2,3))
    std = torch.std(outputs**2,dim=(2,3))
    return torch.mean(torch.div(mu**2,std**2))
        
def vor_loss(inputs,output):
    ratio=torch.div(inputs**2,output**2+eps)
    std = torch.std(ratio,dim=(2,3))
    
    loss = torch.mean(std**2)
    return loss

def mor_loss(inputs,output):
    ratio=torch.div(inputs**2,output**2+eps)
    loss = torch.mean(ratio)
    return loss

def moi_loss(inputs,outputs):
    loss = torch.abs(torch.mean(inputs**2)-torch.mean(outputs**2))/(torch.mean(inputs**2)+torch.mean(outputs**2))
    return loss

# reference version (only for supervised methods)
def enl_loss_ref(outputs,ref):
    mu = torch.mean(outputs**2,dim=(2,3))
    std = torch.std(outputs**2,dim=(2,3))
    mu_ref = torch.mean(ref**2,dim=(2,3))
    std_ref = torch.std(ref**2,dim=(2,3))
    return torch.mean(torch.abs(torch.div(mu**2,std**2)-torch.div(mu_ref**2,std_ref**2)))
        
def vor_loss_ref(inputs,output,ref):
    ratio=torch.div(inputs**2,output**2+eps)
    std = torch.std(ratio,dim=(2,3))
    ratio_ref=torch.div(inputs**2,ref**2+eps)
    std_ref = torch.std(ratio_ref,dim=(2,3))
    return torch.mean(torch.abs(std**2-std_ref**2))

def mor_loss_ref(inputs,output,ref):
    ratio=torch.div(inputs**2,output**2+eps)
    ratio_ref=torch.div(inputs**2,ref**2+eps)
    return torch.abs(torch.mean(ratio)-torch.mean(ratio_ref))


