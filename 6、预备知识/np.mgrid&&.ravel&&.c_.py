#np.mgrid[] .ravel() np.c_[] 常一起使用 用于生成网格点
#np.mgrid[起始值：结束值：步长，起始值：结束值：步长，...]
#返回若干组维度相同的等差数组



#x.ravel() 将多维数组变为一维数组，“把.前变量拉直”



#np.c_[]使返回的间隔数值点配对
#np.c_[数组1，数组2，...]
#把x,y的数值配成对

import numpy as np
x,y=np.mgrid[1:3:1,2:4:0.5]
grid =np.c_[x.ravel(),y.ravel()]#为了保证两个数组的维度相同，所以都是两行四列
print("x:",x)
print("y:",y)
print('grid:\n',grid)
