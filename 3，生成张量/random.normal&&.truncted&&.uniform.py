import tensorflow as tf
import numpy as np

a=tf.constant([1,5],dtype=tf.int64)#几个括号就是几阶张量
print(a)
print(a.dtype)
print(a.shape)

#将numpy的数据类型转换为Tensor数据类型
b=np.arange(0,5)
c=tf.convert_to_tensor(a,dtype=tf.int64)
print(b)
print(c)

#创建全为0的张量
a1=tf.zeros([2,3])
#创建全为1的张量
b1=tf.ones(4)
#创建全为指定值的张量
c1=tf.fill([2,2],9)
print(a1)
print(b1)
print(c1)

#生成正态分布的随机数，默认均值为0，标准差为1
#(维度,mean=均值,stddev=标准差)
a2=tf.random.normal([3,3],mean=0.5,stddev=1)
print(a2)

#生成截断式正态分布的随机数
#它是从截断的正态分布中输出随机值，虽然同样是输出正态分布，但是它生成的值是在距离均值两个标准差范围之内的，
# 也就是说，在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
b2=tf.random.truncated_normal([3,3],mean=0.5,stddev=1)
print(b2)

#生成均匀分布随机数
#[min,max)前闭后开
c2=tf.random.uniform([2,2],minval=0,maxval=1)#每个数都符合在0到1之间的均匀分布
print(c2)
