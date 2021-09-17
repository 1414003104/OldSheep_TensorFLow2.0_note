#7，损失函数（loss）:预测值y与已知答案y_的差距

#三种常用：1，MSE 2,自定义 3，ce（cross Entropy）交叉熵

#例子：酸奶日销量y,x1,x2是影响日销量的因素
import tensorflow as tf
import numpy as np

SEED=23455 #随机数种子

rdm=np.random.RandomState(seed=SEED)#生成[0，1)之间的随机数
x=rdm.rand(32,2)
y_=[[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in x] #生成噪声[0,1)/10=[0,0.1),减0.05，得-0.05到0.05
x=tf.cast(x,dtype=tf.float32)

w1=tf.Variable(tf.random.normal([2,1],stddev=1,seed=1))

epoch=15000
lr=0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y=tf.matmul(x,w1)
        loss_mse = tf.reduce_mean(tf.square(y-y_))

    grads=tape.gradient(loss_mse,w1)
    w1.assign_sub(lr*grads)#更新参数w1

    if epoch%500==0:
        print("After %d training,w1 is"%(epoch))
        print(w1.numpy(),"\n")

print("Final w1 is:",w1.numpy())


