# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 02:19:15 2021

@author: junyanee
"""

import os, glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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
np.save("kFood_kind_image_data_KerasCNN.npy", xy)

# 전처리한 데이터 로드
x_train, x_test, y_train, y_test = np.load('./kFood_kind_image_data_KerasCNN.npy', allow_pickle = True)
print(x_train.shape)
print(x_train.shape[0])

# 0~1 사이의 값으로 정규화
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print(y_train.shape)
print(y_train.shape[0])

# 제작한 cnn model : C-C-P-D - C-P-D - C-P-D - FC-D-FC-FC-FC

# 신경망 모델 #
cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(image_width,image_height,3)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(32, activation='relu'))
cnn.add(Dense(16, activation='relu'))
cnn.add(Dense(nb_classes, activation='softmax'))


# 모델 구조 확인
cnn.summary()

# 신경망 모델 학습
cnn.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_test, y_test), verbose = 1)

# 신경망 구조, 가중치 저장
cnn.save("kFood_cnn_KerasCNN_2.h5")

# 신경망 모델 정확률 평가
res = cnn.evaluate(x_test, y_test, verbose = 0)
print("accuracy is", res[1] * 100)

# 정확률 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.grid()
plt.show()

# 손실함수 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.grid()
plt.show()