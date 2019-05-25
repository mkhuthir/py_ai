#!/usr/bin/python3

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
mnist = tf.keras.datasets.mnist  
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Use Matplotlib to visualize one record
# I set the colormap to Greys. There are a bunch of
# other colormap choices if you like bright visualizations.

plt.imshow(x_train[55].reshape(28, 28), cmap=cm.Greys)

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
plt.show
