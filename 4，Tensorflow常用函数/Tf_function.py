import tensorflow as tf

x1=tf.constant([1.,2.,3.],dtype=tf.float64)
a1=tf.constant([[1.,2.],[3.,4.]],dtype=tf.float64)
print(x1)
print(a1)

#强制tensor转换为该数据类型
x2=tf.cast(x1,tf.int32)
a2=tf.cast(a1,tf.int32)
print(x2)
print(a2)
#计算张量维度上元素的最小值
print(tf.reduce_min(x2),tf.reduce_max(x2))
print(tf.reduce_min(a2),tf.reduce_max(a2))