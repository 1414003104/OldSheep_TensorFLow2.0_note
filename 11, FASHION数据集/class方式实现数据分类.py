import tensorflow as tf

fashion=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

class FashionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten=tf.keras.layers.Flatten()
        self.d1=tf.keras.layers.Dense(128,activation='relu')
        self.d2=tf.keras.layers.Dense(10,activation='softmax')

    def call(self,x):
        x=self.flatten(x)
        x=self.d1(x)
        y=self.d2(x)
        return y

model=FashionModel()

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test),validation_freq=1)
model.summary()