#池化用于减少卷积神经网络中特征数据量

#最大池化 可提取 图片纹理，均值池化 可保留 背景特征。

#如果用2*2的池化核，步长为2，那么输出图片将变为输入图片的四分之一的大小。

#最大池化是用2*2的池化核框住4个像素点，取最大值
#平均池化是。。。。取平均

'''
tf.keras.layers.MaxPool2D(
pool_size=池化核尺寸，#正方形写核长整数，或（核高h，核宽w)
strides=池化步长，#步长整数，或（h,w) ,默认为pool_size
padding='valid' or 'same' #使用全零填充是“same",不使用是"valid”（默认)
)

tf.keras.layers.AveragePooling2D(
pool_size=池化核尺寸，#正方形写核长整数，或（核高h，核宽w）
strides=池化步长，#步长整数，或（h,w) ,默认为pool_size
padding='valid'or'same'#使用全零填充是“same",不使用是"valid”（默认）
)
'''

#MaxPool2D(pool_size=(2,2),strides=2,padding='same')