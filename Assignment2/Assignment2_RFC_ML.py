#import required tools
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from skimage.color import rgb2lab, rgb2gray, lab2rgb
from skimage.io import imread, imshow
import scikitplot as skplt

#test importing image and print out one
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

#in order to shuffle the data
import random 

#import packages for classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

import sklearn
# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3,
# #method="predict_proba")

#Globale Variable
IMG_SIZE = 25

#image directory
DATADIR = "images"

#subfolders
CATEGORIES = ["cave", "desert", "forest", "icebergs"]

#Test by importing 1 image
image_gs = imread("images/forest/2020-11-05_09.45.51.png", as_gray=True)
img_rgb = imread("images/forest/2020-11-05_09.45.51.png") 

#show the color image
plt.imshow(img_rgb)
plt.show()

#print a grayscale image
fig, ax = plt.subplots(figsize=(9, 16))
imshow(image_gs, ax=ax)
ax.set_title('Grayscale image')
ax.axis('off');

#print the colour imgage, and it's R G B profile images
fig, ax = plt.subplots(1, 4, figsize = (18, 30))
ax[0].imshow(img_rgb) 
ax[0].axis('off')

ax[0].set_title('original RGB')

for i, cmap in enumerate(['Reds','Greens','Blues']):
    ax[i+1].imshow(img_rgb[:,:,i], cmap=cmap) 
    ax[i+1].axis('off')
    ax[i+1].set_title(cmap[0])
    
plt.show()

#See what the image array looks like
print(img_rgb)

i = 1057*1920*4
print(i)

img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
plt.imshow(img_rgb)
plt.show()

print(img_rgb)
print(img_rgb.shape)

#reshape as one row of pixel information (neededd in this format for the RFC)
img_rgb = img_rgb.reshape(IMG_SIZE*IMG_SIZE*4)

#print out the reshaped image to check if it looks correct
print(img_rgb)
print(len(img_rgb))

#import all the image data

data = []

IMG_SIZE = 4   #Resize the image accordingly

def create_data():
    for category in CATEGORIES:  #cave, desert, forest and icebergs

        path = os.path.join(DATADIR,category)  # create path to cave, desert, forest and icebergs
        class_num = CATEGORIES.index(category)  # get the classification  (0='cave', 1='desert', 2='forest', 3='iceberg')
        for img in tqdm(os.listdir(path)):  # iterate over each image per cave, desert, forest and icebergs
            try:
                img_array = imread(os.path.join(path,img))  # convert to array
                
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                new_array = new_array.reshape(IMG_SIZE*IMG_SIZE*4)    #reshape the data to be a 1 row array of pixels
                                                    #needs to be in this form for RFC (I believe)
                               
                data.append([new_array,class_num]) # add this to our data list
                
               
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_data() #run the function

#shuffle data
random.shuffle(data)

#transform data from list to data array
data = np.array(data)

#check the shape of the np.array
print(data)
print(data.shape)

#look at a particular occurence of the data
print(data[222])

#make sure we have i number of elements elements 
print(len(data[53][0]))

#Pull the label information from the shuffled data
y = []
for sample in data[:len(data)]:
    y.append(sample[1])

print(y)

#pull the feature information from the data
X = []
for sample in data[:len(data)]:
    X.append(sample[0])

#check to make sure we have feature and label information in X and y
print(X[25])
print(y[25])
print(data[25])

#Plot the image (need to reshape the pixel information for this)
temp = X[10]
plt.imshow(temp.reshape(IMG_SIZE,IMG_SIZE,4))
plt.show()

#convert the label and feature iformation from List to numpy array
y = np.array(y)
X = np.array(X)

#check we still have 223 images with 2500 pixels
print(X.shape)
print(y.shape)
print(len(X[0]))

#Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#check the number per split
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

#train the RFC
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=4)
rfc.fit(X_train, y_train)

#test the RFC
rfc.score(X_test, y_test)
y_pred=rfc.predict(X_test)

#Print out a confusion matrix
plt.figure(figsize=(12,8))
f,ax=plt.subplots(1,1,figsize=(12,12))
#sklearn.metrics.plot_confusion_matrix(rfc,y_test, y_pred)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize='true',ax=ax)
plt.show()

plt.figure(figsize=(12,8))
f,ax=plt.subplots(1,1,figsize=(12,12))
# print("Classification report for classifier %s:\n%s\n"
#       % (rfc, metrics.classification_report(y_test, y_pred)))
disp = metrics.plot_confusion_matrix(rfc, X_test, y_test,normalize='true',ax=ax)

train_samples = 5000
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(
    C=50. / train_samples, penalty='l1', solver='saga', tol=0.1
)
clf.fit(X_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(4):
    l1_plot = plt.subplot(1, 4, i+1)
    l1_plot.imshow(coef[i].reshape(IMG_SIZE,IMG_SIZE,4), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')

#run_time = time.time() - t0
#print('Example run in %.3f s' % run_time)
plt.show()