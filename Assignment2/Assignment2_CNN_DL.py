import os
import pickle

# resize, read and convert to 3 channel BGR color image
from cv2 import resize, imread, IMREAD_COLOR

#just to store the image data in numpy array form when read back using pickle
from numpy import array, shape

# progress bar
from tqdm import tqdm

# to shuffle the data so it won't be classes right next to each other
from random import shuffle

# splitting the training data into seperate
from sklearn.model_selection import train_test_split
import numpy
from pickle import load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import metrics
import scikitplot


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
        image_array = resize(imread(os.path.join(label_folder, image), IMREAD_COLOR), (image_resize_size, image_resize_size))
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# the numpy array object is stored in a file called features with no extension
po = open("training/features", "wb")
pickle.dump(x_train, po)
po.close

# the labels list object is also stored in a file
po = open("training/labels", "wb")
pickle.dump(y_train, po)
po.close()

po = open("testing/features", "wb")
pickle.dump(x_test, po)
po.close()

po = open("testing/labels", "wb")
pickle.dump(y_test, po)
po.close()




x = load(open("training/features", "rb")) # numpy array containing information about images in the data set
y = load(open("training/labels", "rb")) # 0-3 labels for which class the image data belongs to
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
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

print(model.summary())

history = model.fit(x, y, batch_size=32, epochs=10, validation_split=0.2)

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('cnn_model.h5')

x_test = pickle.load(open('testing/features', 'rb'))
y_test = pickle.load(open('testing/labels', 'rb'))

x_test = x_test / 255.0 # because this is the format that the data was trained on
y_test = numpy.array(y_test)

model = tf.keras.models.load_model('cnn_model.h5')

y_pred = model.predict(x_test, verbose=1, use_multiprocessing=True)
print(y_pred)

# round numbers so that it can be compared to the labels
rounded_labels = numpy.argmax(y_pred, axis=1) 
print(rounded_labels)

scikitplot.metrics.plot_confusion_matrix(y_test, rounded_labels, normalize='true')
plt.show()

# f1 
print(metrics.f1_score(y_test, rounded_labels, labels=None, pos_label=1, average='weighted', sample_weight=None, zero_division='warn'))