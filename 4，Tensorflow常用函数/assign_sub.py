#赋值操作，更新参数的值并返回。

#调用asssign_sub前，先用tf.Variable定义变量w为可训练（可自更新）。

#w.assign_sub(w要自减的内容）

import tensorflow as tf
w=tf.Variable(4)
w.assign_sub(1) #w = w - 1
print(w)



