import numpy as np
import matplotlib.pyplot as plt

#sigmoid函数
x=np.linspace(-5,5)
y_sigmoid=1/(1+np.exp(-x))
plt.plot(x,y_sigmoid)#坐标轴
plt.grid(True)#网格
plt.show()

#tanh函数
x=np.linspace(-5,5)
y_tanh=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
plt.plot(x,y_tanh)
plt.grid(True)
plt.show()

#ReLU函数
x=np.linspace(-5,5)
y_relu=np.array([0 if item<0 else item for item in x])
plt.plot(x,y_relu)
plt.grid(True)
plt.show()

