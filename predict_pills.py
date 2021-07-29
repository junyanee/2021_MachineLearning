# -*- coding: utf-8 -*-
"""
@author: junyanee

새로운 영상 예측
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.image as img
import matplotlib.pyplot as plt

# 이미지 전처리하는 함수
def Dataization(img_path):
    image_w = 32
    image_h = 32
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)

image_dir= './test_kFood'

# 출력층 클래스
categories = ['센트롬','오메가3','타이레놀','루테인']

src = []
name = []
test = []

# 경로 돌면서 이미지 전처리
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):
        src.append(image_dir +"/"+ file)
        name.append(file)
        test.append(Dataization(image_dir +"/"+ file))
        
        # 이미지 보여주기
        ndarray = img.imread(image_dir +"/"+ file)
        plt.title(file)
        plt.imshow(ndarray)
        plt.show()
        
    

# 저장된 모델 가져와서 예측
test = np.array(test)
model = load_model('pills_params.h5')
predict = model.predict_classes(test)

for i in range(len(test)):
    print(name[i], "사진은", (categories[predict[i]]), "(으)로 판별됩니다.")

