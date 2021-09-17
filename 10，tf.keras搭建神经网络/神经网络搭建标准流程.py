#六步法
'''
1,import

2,train test 告知要喂入网络的训练集和测试集是什么

3,model=tf.keras.models.Sequential 在这里搭建网络结构，逐层描述每层网络，相当于走了一遍前向传播

4,model.compile 在compile中配置训练方法，告诉训练时选择哪种优化器,选择哪种损失函数，选择哪种评测指标

5,model.fit 在fit中告知训练集和测试集的输入特征和标签，告知每个batch是多少，告知要迭代多少次数据集

6,model.summary 打印网络结构和参数统计

'''

