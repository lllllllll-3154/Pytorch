# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:27:10 2021

@author: Use'r
"""

#15整除fizzbuzz，5buzz，3fizz

def fizzbuzz_encode(i):
    if i%15 == 0: 
        return 3
    elif i%5 ==0: 
        return 2
    elif i%3 ==0:
        return 1
    else:
        return 0
    
def fizzbuzz_decode(i,prediction):
    return [str(i),'fizz','buzz','fizzbuzz'][prediction]

def helper(i):
    print(fizzbuzz_decode(i,fizzbuzz_encode(i)))
    
for i in range(1,16):
    helper(i)
    
#数据集
import numpy as np
import torch

NUM_OF_DIGIT = 10
def binary_encode(i,num_digit):
    return np.array([i>>d & 1 for d in range(num_digit)][::-1])

trX = torch.Tensor([binary_encode(i, NUM_OF_DIGIT) for i in range(101,2**NUM_OF_DIGIT)])
trY = torch.LongTensor([fizzbuzz_encode(i) for i in range(101,2**NUM_OF_DIGIT)])
    
#模型
NUM_OF_H =100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_OF_DIGIT, NUM_OF_H),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_OF_H,4)
    )    

#loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)


if torch.cuda.is_available():
    model = model.cuda()
#测试
BATCH_SIZE =128
EPOCH=10000
for i in range(EPOCH):
    for start in range(0,len(trX),BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]
        
        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        
        y_pred = model(batchX) #forward
        loss = loss_fn(y_pred,batchY)
        print('EPOCH ',i,loss.item())
        
        optimizer.zero_grad()
        loss.backward() #backward
        optimizer.step() #gradient descent
        
#测试        
testX = torch.Tensor([binary_encode(i, NUM_OF_DIGIT) for i in range(1,101)])
if torch.cuda.is_available():
    testX = testX.cuda()
    with torch.no_grad(): #测试不需要grad，节约内存
        testY = model(testX)


predicts = zip(range(1,101),testY.max(1)[1].cpu().data.tolist())
print([fizzbuzz_decode(i,x) for i,x in predicts])




