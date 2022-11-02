import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

DIRECTORY = 'C:\DEV\Cat-Dog-Recognition\dataset'
CATEGORIES = ['cats', 'dogs']

IMG_SIZE = 100

data = []


# PUTTING IMAGES INTO VARIABLE DATA FOR PROCESSING
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])

random.shuffle(data)



x = []
y = []

for features, labels in data:
    x.append(features)
    y.append(labels)


    pickle.dump(x, open('X.pkl', 'wb'))
    pickle.dump(y, open('Y.pkl', 'wb'))