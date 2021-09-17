import tensorflow as tf

#tf.Variable()将变量标记为“可训练”，被标记的变量会在反向传播中，记录梯度信息。神经网络训练中，常用该函数标记待训练参数。
w=tf.Variable(tf.random.normal([2,2],mean=0,stddev=1))
#这样在反向传播中就可以通过梯度下降来更新参数W了
