#!/usr/bin/python3
# Muthanna Alwahash
# Feb 2024

# Video explaining this example
# https://www.youtube.com/watch?v=bemDFpNooA8

import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Load the Fashion MNIST data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# show data shapes
print("\n\n")
print("Training Data input shape  :",training_images.shape)
print("Training Data output shape :",training_labels.shape)
print("Testing Data input shape   :",test_images.shape)
print("Testing Data output shape  :",test_labels.shape)

# Use Matplotlib to visualize one record
plt.imshow(training_images[42].reshape(28, 28), cmap=cm.Greys)
plt.show()

# Plot a bunch of records to see sample data
images = training_images[0:18]
fig, axes = plt.subplots(3, 6, figsize=[9,5])

for i, ax in enumerate(axes.flat):
    ax.imshow(training_images[i].reshape(28, 28), cmap=cm.Greys)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# Show distribution of training data labels
counts = np.bincount(training_labels)
nums = np.arange(len(counts))
plt.bar(nums, counts)
plt.show()

# Normalize data (0-255) > (0-1)
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Design the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),       # input images are 28x28 pixels
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # hidden layers
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # output are 10 classes (fashion types)
])   

# Compile the model
model.compile(optimizer = 'adam',
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])

# Train the model
print("\n\n")
print("Fitting Model...")
model.fit(training_images, training_labels, epochs=5)

# Test the model
print("\n\n")
print("Testing model...")
model.evaluate(test_images, test_labels)

# Use the model to predict
print("\n\n")
print("Using model...")
classifications = model.predict(test_images)

# Select one test sample and show it
i = 1

print("Predictions : "  , classifications[i])
print("Predicted No. : ", np.argmax(classifications[i]))

plt.imshow(test_images[i].reshape(28, 28), cmap=cm.Greys)
plt.show()