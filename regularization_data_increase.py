# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:45:08 2021

@author: junyanee
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# CIFAR-10의 부류 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32'); x_train /= 255
x_train = x_train[0:12,]; y_train = y_train[0:12,]

# 앞 12개 영상을 그려줌
plt.figure(figsize = (16, 2))
plt.suptitle("First 12 images in the train set")
for i in range(12):
    plt.subplot(1, 12, i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(y_train[i])])
    
# 영상 증대기 생성
batch_siz = 6  # 한번에 생성하는 양
generator = ImageDataGenerator(rotation_range = 30.0, width_shift_range = 0.2, height_shift_range = 0.2, horizontal_flip = True)
gen = generator.flow(x_train, y_train, batch_size = batch_siz)

# 첫 번째 증대하고 그리기
img, label = gen.next()
plt.figure(figsize = (16, 3))
plt.suptitle("Generator trial 1")
for i in range(batch_siz):
    plt.subplot(1, batch_siz, i+1)
    plt.imshow(img[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(label[i])])
    
# 두 번째 증대하고 그리기
img, label = gen.next()
plt.figure(figsize = (16, 3))
plt.suptitle("Generator trial 2")
for i in range(batch_siz):
    plt.subplot(1, batch_siz, i+1)
    plt.imshow(img[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(label[i])])
    