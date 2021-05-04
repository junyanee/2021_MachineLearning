# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:38:08 2021

@author: junyanee
"""

import tensorflow as tf

# (1) OR 데이터 구축 (입출력)
x = [[0.0, 0.0],[0.0, 1.0],[1.0, 0.0],[1.0, 1.0]]
y = [[-1], [1], [1], [1]]

# (2) 퍼셉트론 구조
w = tf.Variable([[1.0],[1.0]])
b = tf.Variable(-0.5)

# (3) 퍼센트론 동작식
s = tf.add(tf.matmul(x, w), b)
o = tf.sign(s)

print(o)