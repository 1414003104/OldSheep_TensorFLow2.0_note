'''
model.compile(optimizer=优化器，loss=损失函数，metrics=["准确率"])

##Optimizer可选：
'sgd' or tf.keras.optimizers.SGD(Ir=学习率,momentum=动量参数)

‘adagrad' or tf.keras.optimizers.Adagrad (lr=学习率)

‘adadelta' or tf.keras.optimizers.Adadelta (lr=学习率)

'adam' or tf.keras.optimizers.Adam (lr=学习率, beta_1=0.9, beta_2=0.999)

##loss可选:
'mse' or tf.keras.losses.MeanSquaredError()

'sparse_categorical_crossentropy or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) #交叉熵
#比如如果你的输出结果经过了概率分布，就用False，比如用了softmax，这里就设置为False

##Metrics可选:
‘accuracy : y_和y都是数值，如y_=[1] y=[1]

'categorical_accuracy': y_和y都是独热码(概率分布)，如y_=[0,1,0] y=[0.256,0.695,0.048]

'sparse_categorical_accuracy': y_是数值，y是独热码（概率分布），如y_=[1] y=[0.256,0.695,0.048]


'''