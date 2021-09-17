#增大数据量
'''
image_gen train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale =所有数据将乘以该数值
    rotation_range =随机旋转角度数范围
    width_shift_range=随机宽度偏移量
    height_shift_range =随机高度偏移量
    水平翻转:horizontal_flip=是否随机水平翻转 False or True
    随机缩放:zoom_range =随机缩放的范围[1-n,1+n]) 0.5代表将图像随机缩放50%
image gen_train.fit(x_train) #输入一个四维数据

x_train=x_train.reshape(x_train.shape[0],28,28,1)
(60000,28,28)--->(60000,28,28,1)#最后一维是通道数，比如rgba四个数值表示，或者一个灰度值表实，
故统一用一个数组表示，相比于原来的数值标量，就等于增加了一个维度

model.fit（x_train,y_train,batch_size=32,...）
model.fit(image_gen_train.flow(x_train,y_train,batch_size=32),...)
'''

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度，使数据和网络结构匹配 (60000,28,28,1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(x_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test),
          validation_freq=1)
model.summary()