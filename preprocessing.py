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