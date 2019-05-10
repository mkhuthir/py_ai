#!/usr/bin/python3

# Code to visualize training history
# example is a small network to model the Pima Indians onset of diabetes binary classification problem

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# The following is to suppress TensorFlow Warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# load pima indians dataset
print('\nLoading data from csv file')
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
print('Creating model')
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
print('Compiling Model')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
print('Fitting Model')
history = model.fit(X, Y, validation_split=0.33, epochs=200, batch_size=20, verbose=0)

# list all data in history
print('History:',history.history.keys())

# Print model summery
print(model.summary())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
