# -*- coding:utf-8 -*-
# Author：余时锐
# Date: 2020-09-09
# Message：yushirui_01_tensorflow2.0

# tf
import tensorflow as tf
# 神经网络层
from tensorflow import keras
# 矩阵
import numpy as np
# 绘图
import matplotlib.pyplot as plt

# 打印tf版本
print(tf.__version__)

# 导入Fashion MNIST数据集
fashion_mnist = keras.datasets.fashion_mnist

# 加载数据集，返回4个NumPy数组
# train_images和train_labels阵列的训练集
# 图像28x28 NumPy数组，像素值范围是0到255
# 标签是整数数组，范围0到9
# 对应于图像表示的衣服类别
# 0	T恤/上衣
# 1	裤子
# 2	拉过来
# 3	连衣裙
# 4	涂层
# 5	凉鞋
# 6	衬衫
# 7	运动鞋
# 8	袋
# 9	脚踝靴
# 训练集、训练集标签、测试集、测试集标签 = Fashion MNIST数据集对象.加载数据集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 每个图像，映射到一个标签
# 由于类名不包含在数据集中，将类名存在此处，以后绘图时用
# 类名列表
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 训练集形状，60000张图，28*28像素
r = train_images.shape
print(r)
# (60000, 28, 28)

# 训练集标签数量
r = len(train_labels)
print(r)
# 60000

# 训练标签
r = train_labels
print(r)
# [9 0 0 ... 3 0 5]

# 测试集形状
r = test_images.shape
# 打印
print(r)
# (10000, 28, 28)

# 测试集标签数量
r = len(test_labels)
print(r)
# 10000

# 预处理数据
# 训练神经网络前，必须对数据进行预处理
# 如果检查训练集中的第一张图像，看到像素值在0到255之间
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# 预处理
# 归一化
# 数据输入神经网络模型之前，输入数据缩放到0到1的范围
# 将值除以255。相同的方式处理训练集和测试集，非常重要
train_images = train_images / 255.0
test_images = test_images / 255.0


# 验证数据的格式正确，准备好构建和训练网络
# 显示训练集中的前25个图像，并在每个图像下方显示类别名称
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# 建立模型
# 建立神经网络，配置模型的各层，然后编译模型

# 神经网络层
# 神经网络的基本组成部分是层
# 深度学习的大部分内容是将简单的层链接在一起
# 大多数层（如tf.keras.layers.Dense）具有在训练期间的学习的参数

# 神经网络模型 = 神经网络层组合
model = keras.Sequential([
    # 图像格式从二维数组（28 x 28像素）转换为一维数组（28 * 28 = 784像素）
    # 看作是堆叠图像中的像素行并对齐它们
    # 该层没有学习参数。只会重新格式化数据
    keras.layers.Flatten(input_shape=(28, 28)),
    # 像素展平后，网络由tf.keras.layers.Dense两层序列组成
    # 是紧密连接或完全连接的神经层
    # 第一Dense层有128个节点（神经元）
    keras.layers.Dense(128, activation='relu'),
    # 第二层（也是最后一层）返回长度为10的logits数组
    # 每个节点包含一个得分，该得分指示当前图像属于10个类之一
    keras.layers.Dense(10)
])


# 编译模型
# 准备训练模型前，其它的设置
# 损失函数 -衡量训练过程中模型的准确性。最小化此功能，在正确的方向上“引导”模型
# 优化器 -基于模型看到的数据及其损失函数来更新模型的方式
# 指标 -用于监视培训和测试步骤。以下使用precision，即正确分类的图像比例。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
# 将训练数据输入模型。在此示例中，训练数据在train_images和train_labels数组中。
# 该模型学习关联图像和标签。
# 您要求模型对测试集进行预测（在此示例中为test_images数组）。
# 验证预测是否与test_labels阵列中的标签匹配。
# 喂模型
# 要开始训练，请调用该model.fit方法，之所以这么称呼是因为该方法使模型“适合”训练数据
model.fit(train_images, train_labels, epochs=10)


# 评估准确性
# 接下来，比较模型在测试数据集上的表现：
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# 313/313 - 0s - loss: 0.3354 - accuracy: 0.8811
# 313/313-0s-损耗：0.3354-精度：0.8811
# Test accuracy: 0.8810999989509583
# 测试精度：0.8866000175476074


# 事实证明，测试数据集的准确性略低于训练数据集的准确性
# 训练准确性和测试准确性之间的差距代表过度拟合
# 当机器学习模型在新的，以前看不见的输入上的表现比训练数据上的表现差时，就会发生过度拟合
# 过度拟合的模型“记忆”训练数据集中的噪声和细节，从而对新数据的模型性能产生负面影响。

# 证明过度拟合
# 防止过度拟合的策略


# 作出预测
# 通过训练模型，您可以使用它来预测某些图像。模型的线性输出logits。
# 附加一个softmax层，以将logit转换为更容易解释的概率。

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

# 在这里，模型已经预测了测试集中每个图像的标签。让我们看一下第一个预测：

r=predictions[0]
# 打印
print(r)
# [3.2869505e-07 3.7298345e-11 2.1449464e-11 6.6995610e-11 3.0250091e-11
#  1.1169181e-03 9.4734254e-08 4.4882861e-03 1.9813331e-09 9.9439430e-01]
# 预测是由10个数字组成的数组。它们代表模型对图像对应于10种不同服装中的每一种的“信心”
# 您可以看到哪个标签的置信度最高：


r=np.argmax(predictions[0])
# 打印
print(r)
# 9



r=test_labels[0]
# 打印
print(r)
# 9

# 因此，模型最有信心该图像是脚踝靴或class_names[9]。检查测试标签表明此分类是正确的：




# 完整的10个类预测。

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# 验证预测
# 通过训练模型，您可以使用它来预测某些图像。
#
# 让我们看一下第0张图片，预测和预测数组。正确的预测标签为蓝色，错误的预测标签为红色
# 该数字给出了预测标签的百分比（满分为100）。

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# 绘制一些带有预测的图像。请注意，即使非常自信，该模型也可能是错误的。

# Plot the first X test images, their predicted labels, and the true labels
# Color correct predictions in blue and incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# 使用训练有素的模型
# 最后，使用经过训练的模型对单个图像进行预测。

# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)
# (28, 28)


# tf.keras对模型进行了优化，可以一次对一批或一批示例进行预测。
# 因此，即使您使用的是单个图像，也需要将其添加到列表中：
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)
# (1, 28, 28)

# 为该图像预测正确的标签：
predictions_single = probability_model.predict(img)
print(predictions_single)
# [[1.9319392e-05 3.8494370e-16 9.9736029e-01 3.8851797e-10 2.1668444e-03
#   3.0755891e-13 4.5363593e-04 1.9641396e-20 3.8062348e-10 1.8812749e-17]]
# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)

# keras.Model.predict返回列表列表-数据批次中每个图像的一个列表。
# 批量获取我们（仅）图像的预测：
r = np.argmax(predictions_single[0])
# 打印
print(r)
# 2


# 该模型将按预期预测标签。
#
#
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# 本指南使用Fashion MNIST数据集，其中包含10个类别的70,000个灰度图像。
# 图像显示了低分辨率（28 x 28像素）的单个衣​​物，如下所示：
#
# 时尚MNIST精灵
# 图1. Fashion-MNIST示例（由Zalando，MIT许可）。
#
# Fashion MNIST旨在替代经典MNIST数据集-通常用作计算机视觉的机器学习程序的“ Hello，World”。
# MNIST数据集包含手写数字（0、1、2等）的图像，格式与您将在此处使用的衣服的格式相同。
#
# 本指南将Fashion MNIST用于各种用途，因为它比常规MNIST更具挑战性。
# 这两个数据集都相对较小，用于验证算法是否按预期工作。它们是测试和调试代码的良好起点。
#
# 在这里，使用60,000张图像来训练网络，使用10,000张图像来评估网络学习对图像进行分类的准确度。
# 您可以直接从TensorFlow访问Fashion MNIST。直接从TensorFlow导入和加载Fashion MNIST数据：