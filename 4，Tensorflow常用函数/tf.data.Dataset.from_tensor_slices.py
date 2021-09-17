import tensorflow as tf

#切分传入张量的第一维度，生成输入特征/标签对，构建数据集

#data=tf.data.Dataset.from_tensor_slices((输入特征，标签))
#（Numpy和Tensor格式都可以用该语句读入数据）

features=tf.constant([12,23,10,17])
labels=tf.constant([0,1,1,0])
#特征是12 23 10 17 分别对应的标签是0 1 1 0

dataset=tf.data.Dataset.from_tensor_slices((features,labels))
print(dataset)
for element in dataset:
    print(element)