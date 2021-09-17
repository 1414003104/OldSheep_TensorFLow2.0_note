import tensorflow as tf

#加减乘除
#只有相同维度的张量才可以做四则运算
a=tf.ones([1,3])
#填充
b=tf.fill([1,3],3.)
print(a)
print(b)
print(tf.add(a,b))
print(tf.subtract(a,b))
print(tf.multiply(a,b))
print(tf.divide(a,b))

#平方、次方、开方
#tf.square(张量名)
#tf.pow(张量名，n次方数)
#tf.sqrt(张量名)
c=tf.fill([1,3],3.)
print(tf.pow(c,3))
print(tf.square(c))
print(tf.sqrt(c))

#矩阵乘 tf.matmul
#实现两个矩阵的相乘tf.matmul(矩阵1，矩阵2)
a=tf.ones([3,2])
b=tf.fill([2,3],3.)
print(tf.matmul(a,b))