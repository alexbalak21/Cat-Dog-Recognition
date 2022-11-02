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
        print (img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])

random.shuffle(data)



x = []
y = []
i = 0
for features, labels in data:
    i += 1
    print ('STORE', i)
    x.append(features)
    y.append(labels)


    pickle.dump(x, open('X.pkl', 'wb'))
    pickle.dump(y, open('Y.pkl', 'wb'))