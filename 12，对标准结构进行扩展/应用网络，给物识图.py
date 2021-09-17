#前向传播执行应用

#predict(输入特征，batch_size=整数)
#返回前向传播计算结果

# 复现模型（前向传播）：model=tf.keras.models.Sequential([...]

# 加载参数：model.load_weights(model_save_path)

# 预测结果：result=model.predict(x_predict)

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

model.load_weights(model_save_path)#加载参数

preNum = int(input("input the number of test pictures:"))

for i in range(preNum):
    image_path = input("the path of test picture:")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)#输入的是任意图片，需要resize成为28*28的灰度图片
    img_arr = np.array(img.convert('L'))

    #这里让灰度值小于200变为255，纯白色，让灰度值大于200的变为0，纯黑色。
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]#数据都是按照batch送入网络，添加一个维度
    result = model.predict(x_predict)

    pred = tf.argmax(result, axis=1)#输出最大的概率与测值

    print('\n')
    tf.print(pred)