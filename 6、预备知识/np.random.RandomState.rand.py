import numpy as np

#rand=np.np.random.RandomState.rand(维度)
#返回一个[0,1）之间的随机数
#如果维度为空，返回一个标量

rdm=np.random.RandomState(seed=1)#seed=常数 每次生成的随机数相同
a=rdm.rand() #返回一个随机标量
b=rdm.rand(2,3) #返回维度为2行3列随机数矩阵
c=rdm.rand(3)
d=rdm.rand(1,3)

print("a:",a)
print("b:",b)
print("c:",c)
print("d:",d)
