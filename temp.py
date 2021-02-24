# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#这个不行
x = torch.tensor(5,3,1)
#这个行，类似一个array
x = torch.tensor([5,3,1])
#5x3的1
y = x.new_ones(5,3,dtype = torch.float)
y
#不存在
x.shape()
#3
x.shape
y.shape#5，3
x.size()#3
z = torch.randn_like(5,3)#不行
z = torch.randn_like((5,3))#不行
z = torch.randn_like(y) #y的5，3 shape
z = torch.randn(5,3)#5x3shape

z

x1 = torch.randn_like(x)
x1 = torch.randn_like(y)
x1
y
x1
x1.numel
x1.numel()#元素个数 number of element
x
x.numel()
x
"""
sparse coo 操作
torch.sparse_coo_tensor(1,1)
torch.sparse
torch.sparse()
i = torch.tensor([[0, 1, 1],
                      [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
torch.sparse_coo_tensor(i, v, [2, 4])
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(i, v, [2, 4])
x
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([3, 4], dtype=torch.float32)
x = torch.sparse_coo_tensor(i, v, [2, 4])
x
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([3, 4], dtype=torch.float32)
values
x = torch.sparse_coo_tensor(i, v, [2, 4])
x = torch.sparse_coo_tensor(indices, values, [2, 4])
torch.cuda.is_available()
"""

#GPU选择
if torch.cuda.is_available():
    device = torch.device('cuda') #nvidia gpu
    y = torch.ones_like(x,device = device) # 转换到gpu计算方法1
    x = x.to(device) #转换方法2
    z = x+y
    print(z)
    print(z.to("cpu"),torch.double)

#nunmpy 只能在cpu上使用
