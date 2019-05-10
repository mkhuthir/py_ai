#!/usr/bin/python3

'''
HelloWorld example using TensorFlow library.
'''

import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
op = tf.constant('Hello, TensorFlow!')

# Start a tf session
sess = tf.Session()

# Run the op
result = sess.run(op)

# The value returned by the constructor represents the output
# of the Constant op.
print(result)

