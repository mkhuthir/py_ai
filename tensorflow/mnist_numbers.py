#!/usr/bin/python3
# Muthanna Alwahash
# Feb 2024

import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

# MNIST is the equivalent Hello World of image analysis. 
# It consists of hand written numbers, 0-9, in 28x28 pixel squares.
# Each gray-scale pixel contains an integer 0-255 to indicate darkness,
# with 0 white and 255 black. There are about 60,000 training records, 
# and about 10,000 test records.

# Load data
mnist = tf.keras.datasets.mnist  
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show data shapes
print("\n\n")
print("Training Data input shape  x_train:",x_train.shape)
print("Training Data output shape y_train:",y_train.shape)
print("Testing Data input shape   x_test :",x_test.shape)
print("Testing Data output shape  y_test :",y_test.shape)


# Use Matplotlib to visualize one record
plt.imshow(x_train[55].reshape(28, 28), cmap=cm.Greys)
plt.show()

# Plot a bunch of records to see sample data
images = x_train[0:18]
fig, axes = plt.subplots(3, 6, figsize=[9,5])

for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].reshape(28, 28), cmap=cm.Greys)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# Show distribution of training data labels
counts = np.bincount(y_train)
nums = np.arange(len(counts))
plt.bar(nums, counts)
plt.show()

# Normalize the data (0-255) > (0.0-1.0) scale for faster processing
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
 
# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit Model
print("Fitting Model...")
model.fit(x_train, y_train, epochs=4)

# Evaluate Model
print("Testing model...")
model.evaluate(x_test, y_test)

# Generate predictions for test set
predictions = model.predict(x_test)

i = 88

print("Predictions : "  , predictions[i])
print("Predicted No. : ", np.argmax(predictions[i]))

plt.imshow(x_test[i].reshape(28, 28), cmap=cm.Greys)
plt.show()
