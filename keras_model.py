# -*- coding: utf-8 -*-
"""
Created on Wed May  5 00:55:38 2021

@author: junyanee
"""

from tensorflow.keras.models import Sequential # Sequential은 층을 한 줄로 쌓는데 사용
from tensorflow.keras.layers import Dense # 완전 연결층
from tensorflow.keras.optimizers import SGD # SGD 옵티마이저

# OR  데이터 구축
x = [[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]]
y = [[-1], [1], [1], [1]]

# 신경망 구조 설계
n_input = 2
n_output = 1
perceptron = Sequential() #Sequential 클래스로 객체를 생성
perceptron.add(Dense(units = n_output, activation = 'tanh', input_shape = (n_input, ),
                     kernel_initializer = 'random_uniform',
                     bias_initializer = 'zeros')) # add 함수로 Dense (완전연결) 층을 쌓음

# 신경망 학습
perceptron.compile(loss = 'mse', optimizer = SGD(learning_rate = 0.1), metrics = ['mse'])
perceptron.fit(x, y, epochs = 500, verbose = 2)

# 학습된 신경망으로 예측
res = perceptron.predict(x)
print(res)