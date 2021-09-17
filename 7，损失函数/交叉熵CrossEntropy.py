#CE:表征两个概率分布之间的距离

#交叉熵越小，两个概率分布越近

#tf.losses.categorical_crossentropy(y_,y)

import tensorflow as tf
loss_ce1=tf.losses.categorical_crossentropy([1,0],[0.6,0.4])
loss_ce2=tf.losses.categorical_crossentropy([1,0],[0.8,0.2])
print("loss_ce1:",loss_ce1)
print("loss_ce2:",loss_ce2)
