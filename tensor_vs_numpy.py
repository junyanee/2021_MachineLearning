# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:17:22 2021

@author: junyanee
"""

import tensorflow as tf
import numpy as np

t = tf.random.uniform([2,3], 0, 1)
n = np.random.uniform(0, 1, [2,3])
print("tensorflow로 생성한 텐서:\n", t, "\n")
print("numpy로 생성한 텐서:\n", n, "\n")

res = t + n #텐서 t 와 ndarray n의 덧셈
print("덧셈 결과:\n", res)