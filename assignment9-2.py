# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:43:12 2021

@author: junyanee
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential # Sequential은 층을 한 줄로 쌓는데 사용
from tensorflow.keras.layers import Dense # 완전 연결층
from tensorflow.keras.optimizers import SGD # SGD 옵티마이저

# OR 데이터 구축
x = [[0.0, 0,0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 
     [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
y = [[1], [1], [1], [-1], [1], [-1], [1], [1]]

n_input = 3
n_hidden1 = 5;
n_hidden2 = 5;
n_output = 1

mlp = Sequential() #Sequential 클래스로 객체를 생성
mlp.add(Dense(units = n_hidden1, activation = 'tanh', input_shape = (n_input, ),
                     kernel_initializer = 'random_uniform',
                     bias_initializer = 'zeros')) # add 함수로 Dense (완전연결) 층을 쌓음
mlp.add(Dense(units = n_hidden2, activation = 'tanh', input_shape = (n_input, ),
                     kernel_initializer = 'random_uniform',
                     bias_initializer = 'zeros'))
mlp.add(Dense(units = n_output, activation = 'tanh', input_shape = (n_input, ),
                     kernel_initializer = 'random_uniform',
                     bias_initializer = 'zeros'))

# 신경망 학습
mlp.compile(loss = 'mse', optimizer = SGD(learning_rate = 0.1), metrics = ['mse'])
mlp.fit(x, y, epochs = 500, verbose = 2)

# 학습된 신경망으로 예측
res = mlp.predict(x)
print(res)
