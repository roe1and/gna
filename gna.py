# pylint: disable=missing-docstring, invalid-name, trailing-whitespace
import os
import cv2
import time
import pprint
import ctypes
import random
import pickle

import numpy as np
from PIL import ImageGrab
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

image_dir = 'c:/temp/gna/train'
categories = ['available', 'busy', 'dnd', 'off', 'offline', 'unknown']

'''
x_in = open('x.pickle', 'rb')
x = pickle.load(x_in)
x_in.close()
y_in = open('y.pickle', 'rb')
y = pickle.load(y_in)
y_in.close()

X = x/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=100, epochs=3, validation_split=0.05)
'''

'''
training_data = []
def create_training_data():
    for i, category in enumerate(categories):
        path = os.path.join(image_dir, category)
        class_num = i
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            training_data.append([img_array, class_num])


create_training_data()
random.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

X = np.array(x).reshape(-1, 16, 16, 3)

x_out = open('x.pickle', 'wb')
pickle.dump(X, x_out)
x_out.close()
y_out = open('y.pickle', 'wb')
pickle.dump(y, y_out)
y_out.close()
'''



'''
time.sleep(1)
grab_coords = [117, 250, 117 + 16, 250 + 16]
img = ImageGrab.grab(bbox=grab_coords) 
grab_fn = "img.jpg" 
img.save(grab_fn)
'''
