# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 05:48:14 2021

@author: junyanee
"""

import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model
 
categories = ['간장게장', '갈비구이', '갈비찜', '갈비탕', '갈치구이', '감자채볶음', '감자탕', '계란말이', '계란찜', '고등어구이']
 
def Dataization(img_path):
    image_w = 28
    image_h = 28
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)

 
src = []
name = []
test = []
image_dir = "D:\project_dataset\test_kFood"
print(1)
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):      
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))
        print(1)
 
 
test = np.array(test)
model = load_model('kFood_cnn.h5')
predict = model.predict_classes(test)
print(1)
 
for i in range(len(test)):
    print(name[i] + " : , Predict : "+ str(categories[predict[i]]))