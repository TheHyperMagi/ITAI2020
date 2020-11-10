import numpy
import os
import cv2
import random
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot
from tqdm import tqdm
import pickle

rootdir = "images"
categories = ['bamboo forest', 'cave', 'coral reef', 'desert', 'forest', 'icebergs', 'jungle', 'ocean floor', 'ravine', 'savanna', 'snowy tundra', 'swamp']
training_data = []

for category in categories:
    path = os.path.join(rootdir, category)
    class_num = categories.index(category)

    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path, img))
        new_array = img_array[0:500, 0:500]
        nn_array = cv2.resize(new_array, (32, 32))
        training_data.append([nn_array, class_num])

random.shuffle(training_data)
x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

pyplot.imshow(x[-10])
pyplot.show()

pickle_out = open("features.pickle","wb") # write binary
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()