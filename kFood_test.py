# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 05:48:14 2021

@author: junyanee
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.image as img
import matplotlib.pyplot as plt

def Dataization(img_path):
    image_w = 32
    image_h = 32
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)

image_dir= './test_kFood'

categories = ['간장게장', '갈비구이', '갈비찜', '갈비탕', '갈치구이', '감자채볶음', '감자탕', '계란말이', '계란찜', '고등어구이']

src = []
name = []
test = []

for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):
        src.append(image_dir +"/"+ file)
        name.append(file)
        test.append(Dataization(image_dir +"/"+ file))
        
        ndarray = img.imread(image_dir +"/"+ file)
        plt.title(file)
        plt.imshow(ndarray)
        plt.show()
        
    


test = np.array(test)
model = load_model('kFood_cnn_VGG16.h5')
predict = model.predict_classes(test)

for i in range(len(test)):

    print(name[i], "사진은", (categories[predict[i]]), "(으)로 판별됩니다.")

