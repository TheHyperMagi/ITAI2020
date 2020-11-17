import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from matplotlib import pyplot
import pickle
import numpy

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