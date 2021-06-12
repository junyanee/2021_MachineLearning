# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 16:47:39 2021

@author: junyanee
"""
import os, glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터셋 위치
image_dir = 'D:\project_dataset/CNN_kFood/'

# 클래스 개수 (아웃풋 개수)
categories = os.listdir(image_dir)
nb_classes = len(categories)
print("CNN_kFood에는 ", nb_classes, "개의 클래스가 있습니다.")
print(categories)

# 이미지 사이즈 값 설정
image_width = 32
image_height = 32

# 어레이 생성
x = []
y = []

# 데이터셋 전처리 #
for idx, cat in enumerate(categories):
    # 원 핫 코드로 변경
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    
    # 이미지 불러오기
    image_dir_detail = image_dir + "/" + cat
    files = glob.glob(image_dir_detail + "/*.jpg")
    print(cat, " 파일 개수 : ", len(files))
    
    # 불러온 이미지 전처리해서 배열로 저장
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)

        x.append(data)
        y.append(label)

# 전처리한 값 어레이에 넣기
x = np.array(x)
y = np.array(y)

# 훈련, 테스트 집합 분리
x_train, x_test, y_train, y_test = train_test_split(x, y)
xy = (x_train, x_test, y_train, y_test)

# npy 파일로 저장
np.save("kFood_kind_image_data_vgg16.npy", xy)