#在一个二维张量或数组中，可以通过调整axis等于0或者1控制执行维度
#axis=0代表跨行（经度，down),而axis=1代表跨列（纬度，across)
#如果不指定axis，则所有元素参与计算
#row行 col列

import tensorflow as tf

x1=tf.constant([[1,2,3],
                [2,2,3]])
print(x1)

#计算张量沿着指定维度的平均值
print(tf.reduce_mean(x1))
print(tf.reduce_mean(x1,axis=1))
print(tf.reduce_sum(x1,axis=1))
