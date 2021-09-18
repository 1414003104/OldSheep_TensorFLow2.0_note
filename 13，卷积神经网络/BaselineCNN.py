'''
搭建一个简单的：
C(核:6*5*5，步长:1，填充:same )
B （Yes）
A（relu)
P(max，核:2*2，步长:2，填充:same)
D（0.2)

Flatten
Dense（神经元:128，激活:relu，Dropout:0.2）
Dense（神经元:10，激活:softmax）
'''
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)#没有省略号，打印全部信息

cifar10=tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

class Baseline(Model):
    def __init__(self):
        super(Baseline,self).__init__()
        self.c1=Conv2D(filters=6,kernel_size=(5,5),padding='same')
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        self.p1=MaxPool2D(pool_size=(2,2),strides=2,padding='same')
        self.d1=Dropout(0.2)
        self.flatten=Flatten()
        self.f1=Dense(128,activation='relu')
        self.d2=Dropout(0.2)
        self.f2=Dense(10,activation='softmax')
    def call(self,x):
        x=self.c1(x)
        x=self.b1(x)
        x=self.a1(x)
        x=self.p1(x)
        x=self.d1(x)
        x=self.flatten(x)
        x=self.f1(x)
        x=self.d2(x)
        y=self.f2(x)
        return y

model=Baseline()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoints_save_path="./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoints_save_path+'.index'):
    print('----载入模型----')
    model.load_weights(checkpoints_save_path)

cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_save_path,
                                               save_weights_only=True,
                                               save_best_only=True)

history=model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test),validation_freq=1,callbacks=[cp_callback])

model.summary()

file=open('./weight.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy)+'\n')
file.close()

#### show ####
acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()#给图像加上图例
plt.subplot(1,2,2)
plt.plot(loss,label='Training loss')
plt.plot(val_loss,label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



