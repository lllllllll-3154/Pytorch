# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:05:26 2021

@author: Use'r
"""

#用numpy实现两层神经网络
import numpy as np
N,D_in,H,D_out = 64,1000,100,10

x = np.random.randn(N,D_in)
y = np.random.randn(N,D_out)

w1 = np.random.randn(D_in,H)
w2 = np.random.randn(H,D_out)

l_rate = 1e-6


for t in range(500):
    #forward pass
    h = x.dot(w1)
    h_relu = np.maximum(0,h)
    y_pred = h_relu.dot(w2)
    
    #mse loss
    loss = np.square(y_pred-y).sum()
    print(t,loss)
    
    #backward pass
    #compute gradient
    grad_y_pred = 2.0 * (y_pred-y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    
    
    #update weights
    w1 -= l_rate * grad_w1
    w2 -= l_rate* grad_w2
    
#用torch
import torch 
 
N,D_in,H,D_out = 64,1000,100,10    
    
x = torch.randn(N,D_in)   
y = torch.randn(N,D_out)

w1 = torch.randn(D_in,H)
w2 = torch.randn(H,D_out)

learning_ratge = 1e-6

for t in range(500):
    #forward pass
    h = x.mm(w1) #matrix multi
    h_relu = h.clamp(min=0) #设置上下限
    y_pred = h_relu.mm(w2)
    
    #loss
    loss = (y-y_pred).pow(2).sum().item() #注意把tensor转换成item
    print(t,loss)
    
    #backward pass
    grad_y_pred = 2*(y_pred-y)
    grad_w2 = h_relu.t().mm(grad_y_pred) #小写t表示转置
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone() #copy变成clone
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)


    #update weights
    w1 -= l_rate * grad_w1
    w2 -= l_rate* grad_w2
    


    