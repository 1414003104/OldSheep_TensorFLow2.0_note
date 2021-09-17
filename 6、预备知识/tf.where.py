#tf.where()
#条件语句 真返回A,假返回B
#tf.where(条件语句,真返回A,假返回B)

import tensorflow as tf

a=tf.constant([1,2,3,1,1])
b=tf.constant([0,1,3,4,5])

# tf.greater(a,b)
# 功能：比较a、b两个值的大小
# 返回值:一个列表,元素值都是true和false

c=tf.where(tf.greater(a,b),a,b) #若a>b 返回a对应位置的元素，否则返回b对应位置的元素
print("c:",c)

