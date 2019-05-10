#!/usr/bin/python3

from keras.models import load_model
import tensorflow as tf

# The following is to suppress TensorFlow Warnings
tf.logging.set_verbosity(tf.logging.ERROR)

model = load_model('models/mnist.hdf5')

print(model.summary())