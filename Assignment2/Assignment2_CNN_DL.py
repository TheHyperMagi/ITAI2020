import os
import pickle

import cv2 

# resize, read and convert to 3 channel BGR color image
from cv2 import resize, imread, IMREAD_COLOR

#just to store the image data in numpy array form when read back using pickle
from numpy import array, shape

# progress bar
from tqdm import tqdm

# to shuffle the data so it won't be classes right next to each other
from random import shuffle

import numpy
from pickle import load

# show image using pyplot
from matplotlib import pyplot

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#CONSTANT for all images in the dataset
image_resize_size = 256

# list to hold all the images and label
training_data = []

classes = ["cave", "desert", "forest", "icebergs"]

# iterate through the folders of classes
for label in classes:
    # get url to path
    label_folder = os.path.join('images', label)
    class_num = classes.index(label)
    for image in tqdm(os.listdir(label_folder)):
        # read the image from the path and then resize it 
        image_array = resize(imread(os.path.join(label_folder, image), cv2.IMREAD_COLOR), (image_resize_size, image_resize_size))
        training_data.append([image_array, class_num])

shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

# change the list to an numpy array
# -1 is w/e the first value is, the 2nd and 3rd values is the shape of those two arrays, the last value is the color space
x = array(x).reshape(-1, image_resize_size, image_resize_size, 3) 
y = array(y)

# the numpy array object is stored in a file called features with no extension
po = open("features", "wb")
pickle.dump(x, po)
po.close

# the labels list object is also stored in a file
po = open("labels", "wb")
pickle.dump(y, po)
po.close()

x = load(open("features", "rb")) # numpy array containing information about images in the data set
y = load(open("labels", "rb")) # 0-3 labels for which class the image data belongs to
x = x/255.0 # normalizes to the range of 0-1

model = Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=x.shape[1:]),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout((0.2)),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout((0.2)),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout((0.2)),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout((0.2)),

    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout((0.2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(4)
])

# the logits is the reason why it doesn't work?
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
              
print(model.summary())

model.fit(x, y, batch_size=32, epochs=10, validation_split=0.2)

loss, acc = model.evaluate(x, y, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# print(model.predict(x))
model.save('cnn_model.h5')

x = pickle.load(open('features', 'rb'))
y = pickle.load(open('labels', 'rb'))

x = x / 255.0 # because this is the format that the data was trained on
y = numpy.array(y)

model = tf.keras.models.load_model('cnn_model.h5')

loss, acc = model.evaluate(x, y, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

sample = x[-13:-1]
labels = y[-13:-1]
fig = pyplot.figure()
mm = []
num = 0

categories = ["cave", "desert", "forest", "icebergs"]

for i in labels:
    if i == 0:
        i = "cave"
    elif i == 1:
        i = "desert"
    elif i == 2:
        i = "forest"
    else:
        i = "icebergs"

for data in sample:
    num += 1
    img_data = data

    z = fig.add_subplot(3,5, num)
    orig = img_data.reshape(256,256, 3)
    data = img_data.reshape(-1, 256, 256, 3)

    mm.append(model.predict(data))
    print(mm[num-1][0][0])
    if mm[num-1][0][0] > 0.49:
        str_label = "cave"
    elif mm[num-1][0][1] > 0.49:
        str_label = "desert"
    elif mm[num-1][0][2] > 0.49:
        str_label = "forest"
    else:
        str_label = "icebergs"

    z.imshow(orig)
    
    pyplot.title(str_label + " " + str(labels[num-1]))

    z.axes.get_xaxis().set_visible(False)
    z.axes.get_yaxis().set_visible(False)

pyplot.show()