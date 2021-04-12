# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:33:04 2021

@author: junyanee
"""

from sklearn import datasets
import matplotlib.pyplot as plt

digit = datasets.load_digits()


plt.figure(figsize=(5,5)) #figure : 한 장의 캔버스에 그림을 그리는 것
plt.imshow(digit.images[0], cmap = plt.cm.gray_r, interpolation = 'nearest')
# imshow : 2D로 이미지를 보여줌

plt.show()
print(digit.data[0])
print("이 숫자는 ", digit.target[0], "입니다.")

# 자세한 것은 matplotlib 라이브러리 참조