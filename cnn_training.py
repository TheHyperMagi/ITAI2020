import numpy
from pickle import load

x = load(open("features", "rb")) # numpy array containing information about images in the data set
y = load(open("labels", "rb")) # 0-3 labels for which class the image data belongs to
x = x/255.0 # normalizes to the range of 0-1
# from numpy import shape
# print(shape(x))
# print(shape(y))

# show image using pyplot
from matplotlib import pyplot
# pyplot.imshow(x[1])
# pyplot.show()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

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