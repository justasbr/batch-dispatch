import tensorflow as tf
from numpy import prod


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def alexnet_inference(images):
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
    lrn1 = conv1
    # with tf.name_scope('lrn1') as scope:
    # lrn1 = tf.nn.local_response_normalization(conv1,
    #                                           alpha=1e-4,
    #                                           beta=0.75,
    #                                           depth_radius=2,
    #                                           bias=2.0)

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
    lrn2 = conv2
    # with tf.name_scope('lrn2') as scope:
    #     lrn2 = tf.nn.local_response_normalization(conv2,
    #                                               alpha=1e-4,
    #                                               beta=0.75,
    #                                               depth_radius=2,
    #                                               bias=2.0)

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
