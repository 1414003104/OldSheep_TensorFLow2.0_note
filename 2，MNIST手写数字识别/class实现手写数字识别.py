import tensorflow as tf
#from tensorflow.keras import Model

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

class MNIST_Model(tf.keras.Model):#这里的Model表示继承了tf的Model类
    def __init__(self):
        super(MNIST_Model,self).__init__()
        self.flatten=tf.keras.layers.Flatten()
        self.d1=tf.keras.layers.Dense(128,activation='relu')
        self.d2=tf.keras.layers.Dense(10,activation='softmax')

    def call(self,x):
        x=self.flatten(x)
        x=self.d1(x)
        y=self.d2(x)
        return y

model=MNIST_Model()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test),validation_freq=1)
model.summary()