import tensorflow as tf
'''
tf.keras.layers.Conv2D(

    filter=卷积核个数,
    
    kernel_size=卷积核尺寸, #正方形写核长整数，或（核高h,核宽w）例：(5,5) or 5
    
    strides=滑动步长, #横纵向相同写步长整数，或者（纵向步长h,横向步长w）,默认1
    
    padding="same or valid", #使用全0填充是same,不使用是valid(默认)
    
    activation="relu" or "sigmoid" or "tanh" or "softmax"等,#如果有BN(批标准化)此处不写
    
    input_shape=(高,宽,通道数) #输入特征图维度，可省略
)
'''

# model=tf.keras.models.Sequential([
#     Conv2D(6,5,padding='valid',activation='sigmoid'),
#     MaxPool2D(2,2),
#     Conv2D(6,(5,5),padding='valid',activation='sigmoid'),
#     MaxPool2D(2,(2,2)),
#     Conv2D(filer=6,kernel_size=(5,5),padding='valid',activation='sigmoid'),
#     MaxPool2D(pool_size=(2,2),strides=2),
#     Flatten(),
#     Dense(10,activation='softmax')
# ])