#该函数用于描述各种网络

#网络结构举例

'''
拉直层：tf.keras.layers.Flatten()

全连接层：tf.keras.layers.Dense(神经元个数，activation="激活函数"，kernel_regularizer=哪种正则化)

activation(字符串给出) 可选：relu softmax sigmoid tanh

kernel_regularizer可选： tf.keras.regularizers.l1() tf.keras.regularizers.l2()


卷积层： tf.keras.layers.Conv2D(filters=卷积核个数，kernel_size=卷积核尺寸,strides=卷积步长， padding="valid"or"same")

LSTM层：tf.keras.layers.LSTM()
'''