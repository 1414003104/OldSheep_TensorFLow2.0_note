#输出先过softmax函数，使得输出结果符合概率分布，再计算y与y_得交叉熵损失函数

import tensorflow as tf
import numpy as np

y_=np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]])
y=np.array([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]])
y_pro=tf.nn.softmax(y)
loss_ce1=tf.losses.categorical_crossentropy(y_,y_pro)
loss_ce2=tf.nn.softmax_cross_entropy_with_logits(y_,y)
print("分部计算结果：",loss_ce1)
print("结合计算结果：",loss_ce2)