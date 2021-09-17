#独热编码：在分类问题中，常用独热编码做标签，标记类别：1表实是，0表示非。

#tf.one_hot()函数将带转换数据，转换为one-hot形式的数据输出。
#tf.ont_hot(待转换数据，depth=几分类）

import tensorflow as tf

classes=3
lables=tf.constant([1,0,2])#输入的元素值最小为0，最大为2 三分类的数据标签分别是0：狗尾 1：杂色 2：弗吉尼亚
                                                                #独热码：（0        1        0  ）代表这是杂色
output=tf.one_hot(lables,depth=classes)
print(output)