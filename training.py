import numpy
import cv2
import random
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot
from tqdm import tqdm
import pickle

pickle_in = open("features.pickle","rb") 
x = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("labels.pickle","rb")
y = pickle.load(pickle_in)
pickle_in.close()

y = numpy.array(y) 
x = numpy.array(x).reshape(-1, 32, 32, 3) # -1 is batch size # 105
x = x/255


model = Sequential()
model.add(Conv2D(52 , (3,3), activation='relu', input_shape=x.shape[1:]))
model.add(MaxPooling2D(2, 2))
# The second convolution
model.add(Conv2D(52 , (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
# The third convolution
model.add(Conv2D(52 , (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
# The fourth convolution
model.add(Conv2D(52 , (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
# # The fifth convolution
model.add(Conv2D(52 , (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
# Flatten the results to feed into a DNN
model.add(Flatten())
# 512 neuron hidden layer
model.add(Dense(512, activation='relu'))
# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics='accuracy')

print(model.summary())

# history = model.fit(x, y, batch_size=8, epochs=32, validation_split=0.3)
# history_dict = history.history
# loss_values = history_dict['loss']


# epochs = range(1, len(history_dict['accuracy']) + 1)

# pyplot.plot(epochs, loss_values, 'bo', label='Training loss')

# pyplot.title('Training and validation loss')
# pyplot.xlabel('Epochs')
# pyplot.ylabel('Loss')
# pyplot.legend()

# pyplot.show()