# -*- coding:utf-8 -*-
# Author：余时锐
# Date: 2020-09-09
# Message：yushirui_01_tensorflow2.0

'''
# 学习网址
# https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh-cn

# 安装
pip3 install tensorflow-gpu==2.3.0 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip3 install tensorflow==2.3.0 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

'''

# tensorflow
import tensorflow as tf

# 载入并准备好 MNIST 数据集。将样本从整数转换为浮点数：
mnist = tf.keras.datasets.mnist

# 训练集、测试集 = mnist数据集加载
# 导入Fashion MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 模型 = 神经网络层
# 将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数：
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

r = model.compile(
    optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)
print(r)

# 训练模型
r = model.fit(x_train, y_train, epochs=5)
print(r)

# 验证模型
r = model.evaluate(x_test,  y_test, verbose=2)
print(r)

# 这个照片分类器的准确度已经达到 98%
# [0.07798629999160767, 0.9761999845504761]

