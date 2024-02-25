#!/usr/bin/python3
# Muthanna Alwahash
# Feb 2024

import tensorflow as tf
print(tf.__version__)

# Load the Fashion MNIST data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize data (0-255) > (0-1)
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile the model
model.compile(optimizer = 'adam',
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=5)

# Test the model
model.evaluate(test_images, test_labels)

# Use the model to predict
classifications = model.predict(test_images)

# print results
print(classifications[0])
print(test_labels[0])
