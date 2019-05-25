#!/usr/bin/python3

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Disable some deprecated error messages
tf.logging.set_verbosity(tf.logging.ERROR)


# MNIST is the equivalent Hello World of image analysis. 
# It consists of hand written numbers, 0-9, in 28x28 pixel squares.
# Each gray-scale pixel contains an integer 0-255 to indicate darkness,
# with 0 white and 255 black. There are about 60,000 training records, 
# and about 10,000 test records.

# Load MNIST Data into a Numpy Array

with open('./data/mnist.pkl', 'rb') as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f, encoding='latin1')

print("\nTraining Data X:",x_train.shape)
print("Training Data Y:",y_train.shape)
print("Testing Data X:",x_test.shape)
print("Testing Data Y:",y_test.shape)


# You can also load it directly from the cloud as follows:
# mnist = tf.keras.datasets.mnist  
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Use Matplotlib to visualize one record
# I set the colormap to Greys. There are a bunch of
# other colormap choices if you like bright visualizations.

plt.imshow(x_train[55].reshape(28, 28), cmap=cm.Greys)
plt.show()

# Plot a bunch of records to see sample data

# Basically, use the same Matplotlib commands above 
# in a for loop to show 18 records from the train set 
# in a subplot figure. We also make the figsize a bit 
# bigger and remove the tick marks for readability.

images = x_train[0:18]
fig, axes = plt.subplots(3, 6, figsize=[9,5])

for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].reshape(28, 28), cmap=cm.Greys)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# Show distribution of training data labels¶

counts = np.bincount(y_train)
nums = np.arange(len(counts))
plt.bar(nums, counts)
plt.show()


# Apply Keras/TensorFlow neural network

# Normalize the data to a (0.0 - 1.0) scale for faster processing
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
print("\n\nFitting Model...\n")
model.fit(x_train, y_train, epochs=4)

# Evaluate Model
model.evaluate(x_test, y_test)

# Generate predictions for test set
predictions = model.predict(x_test)

print("\n\nPredictions: \n",predictions[88])
print("\n\nPredicted No. : \n",np.argmax(predictions[88]))
plt.imshow(x_test[88].reshape(28, 28), cmap=cm.Greys)
plt.show()
