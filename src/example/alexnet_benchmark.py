# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Timing benchmark for AlexNet inference.

To run, use:
  bazel run -c opt --config=cuda \
      models/tutorials/image/alexnet:alexnet_benchmark

Across 100 steps on batch size = 128.

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline
from numpy import prod

FLAGS = None

ELEPHANT = "/home/justasbr/tmp_tensorflow/elephant.jpg"
PANDA = "/home/justasbr/tmp_tensorflow/panda.jpg"


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    """Build the AlexNet model.

    Args:
      images: Images Tensor

    Returns:
      pool5: the last Tensor in the convolutional component of AlexNet.
      parameters: a list of Tensors corresponding to the weights and biases of the
          AlexNet model.
    """
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool1
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool2
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)

    maxpool5 = pool5

    # fc6
    # fc(4096, name='fc6')
    # fc6W = tf.Variable(net_data["fc6"][0])
    # fc6b = tf.Variable(net_data["fc6"][1])
    # print(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))])) #1, 9216
    with tf.name_scope('fcc6') as scope:
        fc6W = tf.Variable(tf.random_normal([9216, 4096],
                                            mean=0,
                                            stddev=1.0,
                                            dtype=tf.float32,
                                            name='fc6_weights'))

        fc6b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32,
                                       name='fc6_biases'))

        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        print_activations(fc6)


        # fc7
        # fc(4096, name='fc7')
    with tf.name_scope('fcc7') as scope:
        fc7W = tf.Variable(tf.random_normal([4096, 4096], mean=0, stddev=1.0, dtype=tf.float32, name='fc7_weights'))
        fc7b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32, name='fc7_biases'))

        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

        print_activations(fc7)

        # fc8
        # fc(1000, relu=False, name='fc8')
    with tf.name_scope('fcc7') as scope:
        fc8W = tf.Variable(tf.random_normal([4096, 1000], mean=0, stddev=1.0, dtype=tf.float32, name='fc8_weights'))
        fc8b = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32, name='fc8_biases'))
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        print_activations(fc8)

    # return fc8, parameters

    prob = tf.nn.softmax(fc8)
    return prob, parameters


def time_tensorflow_run(session, target, info_string):
    """Run the computation to obtain the target tensor and print timing stats.

    Args:
      session: the TensorFlow session to run the computation under.
      target: the target Tensor that is passed to the session's run() function.
      info_string: a string summarizing this run, to be printed with the stats.

    Returns:
      None
    """
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    file_name = 'timeline_alexnet' + str(FLAGS.batch_size) + '.json'
    for i in xrange(FLAGS.num_batches + num_steps_burn_in):
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start_time = time.time()
        probabilities = session.run(target, options=options, run_metadata=run_metadata)
        # print(np.argmax(probabilities))
        # for prob in probabilities:
        #    print(np.argmax(prob), end=" ")
        # print()
        duration = time.time() - start_time

        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()

        with open(file_name, 'w') as f:
            f.write(chrome_trace)

        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_benchmark():
    """Run the benchmark on AlexNet."""
    with tf.Graph().as_default():
        image_size = 224

        # Note that our padding definition is slightly different the cuda-convnet.
        # In order to force the model to start with the same activations sizes,
        # we add 3 to the image_size and employ VALID padding above.

        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once("/home/jb/tmp_tensorflow/*.jpg"))
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        img_decoded = tf.image.decode_jpeg(image_file, channels=3)

        # img_decoded = tf.cast(images, tf.float32)

        img_decoded = tf.image.per_image_standardization(img_decoded)
        img_decoded = tf.clip_by_value(img_decoded, -1.0, 1.0)

        img_batch = tf.train.batch([img_decoded], batch_size=FLAGS.batch_size, shapes=(image_size, image_size, 3),
                                   capacity=10)

        # image_data2 = tf.gfile.FastGFile(PANDA, 'rb').read()
        # print(image_data2)

        # print("IMAGES ", images)
        # images = tf.Variable(tf.random_normal([FLAGS.batch_size, image_size, image_size, 3],
        #                                      dtype=tf.float32, stddev=1e-1))

        print("IMG", img_batch)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        prob, parameters = inference(img_batch)

        # Build an initialization operation.
        local_init = tf.local_variables_initializer()
        global_init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        config = tf.ConfigProto()
        # config.gpu_options.allocator_type = 'BFC'
        print("PRE-SESS")
        sess = tf.Session(config=config)

        sess.run(local_init)
        sess.run(global_init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        print("POST-SESS")

        # Run the forward benchmark.
        time_tensorflow_run(sess, prob, "Forward")
        print("POST-RUN")
        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)


def main(_):
    run_benchmark()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size.'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=25,
        help='Number of batches to run.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
