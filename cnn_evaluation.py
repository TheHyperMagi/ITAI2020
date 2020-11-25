import tensorflow as tf

from matplotlib import pyplot
import pickle
import numpy

from sklearn import metrics
import scikitplot

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
pyplot.show()

# f1 
print(metrics.f1_score(y_test, rounded_labels, labels=None, pos_label=1, average='weighted', sample_weight=None, zero_division='warn'))