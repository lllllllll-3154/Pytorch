# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:05:09 2021

@author: Use'r
"""

import torch
'''
#x = torch.tensor(1.,requires_grad = True)
w=  torch.tensor(2.,requires_grad = True)
b=  torch.tensor(3.,requires_grad = True)

y = w*x + b

y.backward()  #y = 2*x +3


print( w.grad)
print(b.grad)
'''
'''
N,D_in,H,D_out = 64,1000,100,10    
    
x = torch.randn(N,D_in)   
y = torch.randn(N,D_out)

w1 = torch.randn(D_in,H,requires_grad=True)  #requires 
w2 = torch.randn(H,D_out,requires_grad=True)

learning_ratge = 1e-6

for t in range(500):
    #forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    #loss
    loss = (y-y_pred).pow(2).sum() #此时不能转换成item（数字）
    #print(t,loss)
    loss.backward()
    print(t,loss.item())
    #backward pass
    with torch.no_grad(): #节约内存 
    #update weights
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        
        w1.grad.zero_() #清零
        w2.grad.zero_()   

'''
import torch.nn as nn

'''
N,D_in,H,D_out = 64,1000,100,10    
    
x = torch.randn(N,D_in)   
y = torch.randn(N,D_out)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
    )

loss_fn = nn.MSELoss(reduction='sum')

nn.init.normal_(model[0].weight)  #initialization 的关系
nn.init.normal_(model[2].weight)

#model = model.cuda()
for t in range(500):
    #forward pass
    y_pred =model(x)
    #loss
    loss = loss_fn(y_pred,y)
    #print(t,loss)
    loss.backward()
    print(t,loss.item())
    #backward pass
    with torch.no_grad(): #节约内存 
    #update weights
        for param in model.parameters(): #param包含tensor和grad
            param -= learning_rate * param.grad
    
    model.zero_grad()
    
'''    
  
'''
#ADAM  
N,D_in,H,D_out = 64,1000,100,10    
    
x = torch.randn(N,D_in)   
y = torch.randn(N,D_out)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
    )

loss_fn = nn.MSELoss(reduction='sum')

#nn.init.normal_(model[0].weight)  #initialization 的关系
#nn.init.normal_(model[2].weight)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
#model = model.cuda()
for t in range(500):
    #forward pass
    y_pred =model(x)
    #loss
    loss = loss_fn(y_pred,y)
    #print(t,loss)
    optimizer.zero_grad()
    loss.backward()
    print(t,loss.item())
    #backward pass

    optimizer.step()
'''    
    
#自定义nn
N,D_in,H,D_out = 64,1000,100,10    
    
x = torch.randn(N,D_in)   
y = torch.randn(N,D_out)

class TwoLayer(nn.Module):

    def __init__(self, D_in,H,D_out):
        super(TwoLayer,self).__init__()
        self.linear1  = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H,D_out)
        
    def forward(self,x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred

model = TwoLayer(D_in, H, D_out)
loss_fn = nn.MSELoss(reduction='sum')

#nn.init.normal_(model[0].weight)  #initialization 的关系
#nn.init.normal_(model[2].weight)
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
#model = model.cuda()
for t in range(500):
    #forward pass
    y_pred =model(x)
    #loss
    loss = loss_fn(y_pred,y)
    #print(t,loss)
    optimizer.zero_grad()
    loss.backward()
    print(t,loss.item())
    #backward pass

    optimizer.step()
    
class A():
    def __init__(self,name = 'l'):
        self.name = name

class B(A):
    pass

class C(A):
    def __init__(self,age=10):
        self.age = age

class D(A):
    def __init__(self,name='s',age=10):
        self.age = age   
        super().__init__(name)
        