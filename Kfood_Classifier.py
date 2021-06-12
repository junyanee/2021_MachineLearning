# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:41:47 2021

@author: junyanee

kFood class : 15
images per class : 300

"""
import os, glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


image_dir = 'D:\project_dataset/CNN_kFood/'
categories = os.listdir(image_dir)
nb_classes = len(categories)

print(categories)

image_width = 28
image_height = 28

x = []
y = []

for idx, cat in enumerate(categories):
    # one-hot coding
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    
    image_dir_detail = image_dir + "/" + cat
    files = glob.glob(image_dir_detail + "/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)

        x.append(data)
        y.append(label)

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y)
xy = (x_train, x_test, y_train, y_test)
np.save("kFood_kind_image_data.npy", xy)

import os, glob, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = np.load('./kFood_kind_image_data.npy', allow_pickle = True)
print(x_train.shape)
print(x_train.shape[0])

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print(y_train.shape)
print(y_train.shape[0])

# cnn model : LeNet-5

cnn = Sequential()
cnn.add(Conv2D(6,(5,5), padding = 'same', activation = 'relu', input_shape = (28,28,3))) #커널 개수, 사이즈 미입력
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(16,(5,5), padding = 'same', activation = 'relu')) #커널 개수, 사이즈 미입력
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(120,(5,5), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.5))
cnn.add(Flatten())
cnn.add(Dense(84, activation = 'relu'))
cnn.add(Dense(10, activation = 'softmax'))
cnn.summary()

# 신경망 모델 설계
cnn.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_test, y_test))
cnn.save("kFood_cnn.h5")


# 신경망 모델 정확률 평가
res = cnn.evaluate(x_test, y_test, verbose = 0)
print("accuracy is", res[1] * 100)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'best')
plt.grid()
plt.show()