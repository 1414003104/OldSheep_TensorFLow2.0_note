#将两个数组按垂直方向叠加
#np.vstack(数组1，数组2)

import numpy as np

a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.array([[7,8,9],
            [10,11,12]])
d=np.vstack((a,b))
e=np.vstack((a,b,c))
print("c:\n",d)
print("d:\n",e)