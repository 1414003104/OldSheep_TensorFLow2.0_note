import tensorflow as tf
import matplotlib.pyplot as plt

cifar10=tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

plt.imshow(x_train[0]) #绘制图片
plt.show()

print("x_train[0]:\n",x_train[0])

print(y_train)

print(x_test.shape)

