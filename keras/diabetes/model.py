#!/usr/bin/python3

# Code to visualize training history
# example is a small network to model the Pima Indians onset of diabetes binary classification problem

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# The following is to suppress TensorFlow Warnings
#tf.logging.set_verbosity(tf.logging.ERROR)

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
history = model.fit(X, Y, validation_split=0.40, epochs=500, batch_size=10, verbose=0)

# list all data in history
print('History:',history.history.keys())

# Print model summery
print(model.summary())

# Print model accuracy
acc = model.evaluate(X,Y)
print('accuracy:', acc[1] * 100, '%')

# plot model history
f, ax = plt.subplots(1, 2, figsize=(12,6))

# summarize history for accuracy
ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title('model accuracy')
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'test'], loc='upper left')

# summarize history for loss
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'test'], loc='upper left')
plt.show()
