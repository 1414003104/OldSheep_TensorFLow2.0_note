import tensorflow as tf

#with结构记录计算过程，gradient求出张量的梯度
#在tape下面申明可导的变量。如果要对constant求导就要tape.watch()
# with tf.GradientTape() as tape:
#
#     若干个计算过程
# grad=tape.gradient(函数,对谁求导)


with tf.GradientTape() as tape:
    w=tf.Variable(tf.constant(3.0))
    loss=tf.pow(w,2)
grad=tape.gradient(loss,w)
print(grad)