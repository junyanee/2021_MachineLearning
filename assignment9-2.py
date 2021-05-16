# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:43:12 2021

@author: junyanee
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

x = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 
     [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
y = [[1], [1], [1], [-1], [1], [-1], [1], [1]]

n_input = 3
n_hiddne1 = 5
n_hidden2 = 5
n_output =1

perceptron=Sequential()


perceptron.add(Dense(units=n_hiddne1,activation='tanh',input_shape=(n_input,),
                     kernel_initializer='random_uniform',bias_initializer='zeros'))
perceptron.add(Dense(units=n_hidden2,activation='tanh',input_shape=(n_input,),
                     kernel_initializer='random_uniform',bias_initializer='zeros'))
perceptron.add(Dense(units=n_output,activation='tanh',input_shape=(n_input,),
                     kernel_initializer='random_uniform',bias_initializer='zeros'))
perceptron.compile(loss='mse',optimizer=SGD(learning_rate=0.1),metrics=['mse'])

perceptron.fit(x,y,epochs=500,verbose=2)
res=perceptron.predict(x)
print(res)