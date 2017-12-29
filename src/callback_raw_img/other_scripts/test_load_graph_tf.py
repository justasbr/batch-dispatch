'''
HelloWorld example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np


def load_frozen_graph():
    with tf.gfile.GFile("tf_vgg.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="main")
    return graph

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
g = load_frozen_graph()
sess = tf.Session(graph=g)


# Run the op
for op in g.as_graph_def().node:
    print(op.name)

# for op in g.get_operations():
#     print(op.name)

x = g.get_tensor_by_name('main/input_1:0')
y = g.get_tensor_by_name('main/output_node0:0')


for i in range(10):
    print("BEFORE")
    y_out = sess.run(y, feed_dict={x: np.random.randint(225, size=(5,224,224,3))})
    print(y_out.argmax(axis=1))
    # print(sess.run(y))
    print("AFTER")
