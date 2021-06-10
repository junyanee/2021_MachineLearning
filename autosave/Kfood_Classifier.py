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