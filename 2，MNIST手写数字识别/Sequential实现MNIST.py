import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#归一化，使得0到255之间的灰度值变成0到1之间的数值
x_train,x_test=x_train/255.0,x_test/255.0

model=tf.keras.Sequential([
    tf.keras.layers.Flatten(),#拉直为一维数组，也就是拉直成748个数值
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test),validation_freq=1)
model.summary()
