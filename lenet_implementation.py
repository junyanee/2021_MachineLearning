# -*- coding: utf-8 -*-
"""
Created on Thu May 20 09:57:09 2021

@author: junyanee
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# MNIST 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# LeNet-5 neural network model
cnn = Sequential()
cnn.add(Conv2D(6, (5, 5), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(16, (5, 5), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(120, (5, 5), padding = 'same', activation = 'relu'))
cnn.add(Flatten())
cnn.add(Dense(84, activation = 'relu'))
cnn.add(Dense(10, activation = 'softmax'))

