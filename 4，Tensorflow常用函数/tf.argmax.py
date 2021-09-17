
#返回张量指定维度最大值的索引
#tf.argmax(张量名,axis=操作轴)
import tensorflow as tf
import numpy as np

test=np.array([[1,2,3],[2,3,4],[5,4,3],[8,7,2]])
print(test)

print(tf.argmax(test,axis=0)) #返回每一列最大值的索引
print(tf.argmax(test,axis=1)) #返回每一行最大值的索引
