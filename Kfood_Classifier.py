# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:41:47 2021

@author: junyanee

kFood class : 15
images per class : 300

"""

import numpy as np
from PIL import Image
import os

def one_hot(i):
    a = np.zeros(15, 'uint8')
    a[i] = 1
    return a

data_dir = './Images/'
nb_classes = 15

result_arr = np.empty(12198, 12303)

idx_start = 0

for cls, food_name in enumerate(os.listdir(data_dir)):
    image_dir = data_dir + food_name + '/'
    file_list = os.listdir(image_dir)
    
    for idx, f in enumerate(file_list):
        im = Image.open(image_dir + f)
        pix = np.array(im)
        arr = pix.reshape(1, 12288)
        result_arr[idx_start + idx] = np.append(arr, one_hot(cls))
    idx_start += len(file_list)

np.save('result.npy', result_arr)

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = np.load(data_dir)

# cnn model - C-P-D-C-P-C-C-P-D-FC-D-FC-D-FC-D-FC
cnn = Sequential()
cnn.add(Conv2D(32(8,8), activation = 'relu', input_shape = 'dfdfdfdffd')) #커널 개수, 사이즈 미입력
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(5,5)), activation = 'relu')
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(128,(3,3)), activation = 'relu')
cnn.add(Conv2D(128,(3,3)), activation = 'relu')
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(512), activation = 'relu')
cnn.add(Dropout(0.25))
cnn.add(Dense(128), activation = 'relu')
cnn.add(Dropout(0.25))
cnn.add(Dense(64), activation = 'relu')
cnn.add(Dropout(0.5))
cnn.add(Dense(15), activation = 'softmax')

# tlsrudakd ahepf gkrtmq
cnn.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size = 64, epochs = 30, validation_data = (x_test, y_test), verbose = 2)

# tlsrudakd ahepf wjdghkrfbf vudrk
res = cnn.evaluate(x_test, y_test, verbose = 0)
print("accuracy is", res[1] * 100)

import matplotlib.pyplot as plt