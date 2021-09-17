#sequential无法写出一些带有跳连的非顺序网络结构，这时候用class搭建神经网络

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

x_train=datasets.load_iris().data
y_train=datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

#定义模型
class IrisModel(Model):
    def __init__(self):
        # 定义网络结构块
        super(IrisModel,self).__init__()#super() 函数是用于调用父类(超类)的一个方法。
        self.d1=Dense(3,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2())


    def call(self,x):
        #调用网络结构快，实现前向传播
        y=self.d1(x)
        return y

model=IrisModel()#实例化

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train, batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)
model.summary()
