# -*- coding: utf-8 -*-
"""
Created on Sat May 26 23:52:07 2018

@author: ASUS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier=Sequential()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255
number_of_classes = 10
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
classifier.add(Convolution2D(32,3,3,input_shape=(28,28,1),activation='relu'))
classifier.add(BatchNormalization(axis=-1))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(BatchNormalization(axis=-1))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=256,activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(output_dim=10,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
classifier.fit_generator(train_generator, steps_per_epoch=60000, epochs=4, 
                    validation_data=test_generator, validation_steps=10000)
ix = 0
from keras.models import load_model
classifier = load_model('my_model.h5')
classifier.save('my_model.h5') \
import cv2
image = cv2.imread("pitrain.png")

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
gray = cv2.bilateralFilter(thresh1, 11, 17, 17)
kernel = np.ones((5,5), np.uint8)
#img_dilation = cv2.dilate(gray, kernel, iterations=1)
#img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
#edged = cv2.Canny(gray, 30, 200)
plt.imshow(gray[:,:], 'gray')
im2,contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)

roi=[]
i=0
for contour in contours:
 
    
    [x,y,w,h] = cv2.boundingRect(contour)
    
 
    if h>300 and w>300:
 
        continue
    
 
    if h<40 or w<40:
 
        continue
    
    i=i+1
    roi.append(image[y:y+h, x:x+w])
    
im_names=["im1.png","im2.png","im3.png"]
i=0
"""for roi in roi:
    cv2.imwrite(im_names[i], roi)
    i=i+1"""
str1=''
for roi in roi:
    
    im = cv2.bitwise_not(roi[0])
    #im= cv2.copyMakeBorder(roi[0],5,5,5,5,cv2.BORDER_CONSTANT,value=[0,0,0])

    #cv2.imwrite("contoured.png", image)
    from skimage.transform import resize

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    #im = plt.imread("contoured.png")
    im = rgb2gray(im)
    im = resize(im, (28, 28))

    plt.imshow(im[:,:], 'gray')
    #plt.show()

    temp = np.zeros((1,28,28,1))
    temp[0,:,:,0] = im
    num=int(classifier.predict_classes(temp))
    str1+=str((num))
print(str1)
