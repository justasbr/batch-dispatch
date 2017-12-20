import _thread as thread
from concurrent.futures import ThreadPoolExecutor

import rpyc
from rpyc.utils.server import ThreadedServer

import queue
import time

import torch
import torchvision
from torch.autograd import Variable
from alexnet_torch import alexnet
from skimage import io
import argparse
import random

import cProfile, pstats, io as io2

import tensorflow as tf
from numpy import prod
import os
import numpy as np

BATCH_SIZE = 4
BATCH_TIMEOUT = 0.1
FRAMEWORK = ""

tpe = ThreadPoolExecutor(max_workers=4)

image_queue = queue.Queue()
time_q = queue.Queue()

total_latency = 0.0
total_done = 0

global classify
global t1


def prepare_torch():
    global net, transform
    net = alexnet(pretrained=True)
    transform = torchvision.transforms.ToTensor()


def torch_classify(imgs):
    img_data = []
    for img in imgs:
        data = transform(img)
        data = Variable(data)
        img_data.append(data)

    batch = torch.stack(img_data)
    output = net.forward(batch)
    max_val, max_index = torch.max(output, 1)
    output = max_index.data.numpy()

    return output


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


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def prepare_tf():
    global fnames, enq_op, prob, main_sess, images_placeholder
    g = tf.Graph()
    with g.as_default():
        images_placeholder = tf.placeholder(tf.uint8, shape=(None, 224, 224, 3))

        img_batch_float = tf.cast(images_placeholder, tf.float32)
        img_batch_float = tf.map_fn(tf.image.per_image_standardization, img_batch_float)
        img_batch_float = tf.map_fn(lambda frame: tf.clip_by_value(frame, -1.0, 1.0), img_batch_float)
        # image_batch = tf.stack(images)
        # print(image_batch) #/image_batch.dtype)

        prob, __ = alexnet_inference(img_batch_float)

        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
    main_sess = tf.Session(graph=g)
    main_sess.run(init_global)
    main_sess.run(init_local)



def round_ms(seconds):
    return round(1000 * seconds, 3)

def tf_classify(imgs):
    # img_data = np.empty(shape=(len(imgs), 224, 224, 3))
    # for i, img in enumerate(imgs):
    #     print(img)
    # img_data[i] = img
    result = main_sess.run(prob, feed_dict={images_placeholder: imgs})

    return result.argmax(axis=1)


def mock_classify(raw_images):
    def get_random_num(raw_img):
        return 0
        # return random.randint(0, 1000)

    return list(map(get_random_num, raw_images))


class MyService(rpyc.Service):
    def __init__(self, conn):
        super().__init__(conn)
        print("MASTER INIT")

    def on_connect(self):
        """
        Invoked when the client connects.  Performs required initialization
        """
        print("Executive connected", thread.get_ident())

    def on_disconnect(self):
        """
        Invoked when the client disconnects.
        """
        print("Executive has disconnected...")

    def exposed_RemoteCallbackTest(self, raw_img, callback):
        # print("GOT IMG", t1)
        # callback("0")
        img = np.fromstring(raw_img, dtype=np.uint8).reshape(224, 224, 3)
        image_queue.put((img, callback))

        # global t1
        # t1 = time.time()
        time_q.put(time.time())
        # print(img.shape)
        # print("SIZE: ", sys.getsizeof(img))


def parse_arguments():
    global BATCH_SIZE, FRAMEWORK, BATCH_TIMEOUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch_size", type=int)
    parser.add_argument("-t", "--timeout", help="timeout for batch", type=float)
    parser.add_argument("-f", "--framework", help="framework")

    args = parser.parse_args()
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size

    if args.framework is not None:
        FRAMEWORK = args.framework

    if args.timeout is not None:
        BATCH_TIMEOUT = args.timeout

    print("Batch sizes:", BATCH_SIZE)
    print("Batch timeout:", BATCH_TIMEOUT)


def run_job_batcher():
    batch_size = BATCH_SIZE
    last_batch_time = None

    # TODO - now first batch always full
    while True:
        if image_queue.qsize() >= batch_size:
            last_batch_time = time.time()
            process_and_return(batch_size)
        elif image_queue.qsize() and last_batch_time and (time.time() - last_batch_time) >= BATCH_TIMEOUT:
            last_batch_time = time.time()
            print("Timed out")
            process_and_return(image_queue.qsize())
        else:
            time.sleep(0)

def process_and_return(batch_size):
    inputs = [None] * batch_size
    callbacks = [None] * batch_size
    for i in range(batch_size):
        inputs[i], callbacks[i] = image_queue.get()

    classify_start = time.time()
    pr.enable()
    outputs = classify(inputs)
    pr.disable()
    classify_end = time.time()
    print("Classify, batch_size=", batch_size, "took (ms)", round_ms(classify_end-classify_start))

    # s = io2.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    # ps.print_stats()
    # print(s.getvalue())

    for i in range(batch_size):
        global total_latency, total_done
        curr_latency = time.time() - time_q.get()

        total_latency += curr_latency
        total_done += 1

        print("RETURN1 (ms)", round_ms(curr_latency), "AVG (ms)", round_ms(total_latency / total_done))
        callbacks[i](outputs[i])
        # tpe.submit(callbacks[i], outputs[i])


def prepare_framework():
    global classify
    if FRAMEWORK == "tf" or FRAMEWORK == "tensorflow":
        prepare_tf()
        classify = tf_classify
        print("Using TensorFlow")
    elif FRAMEWORK == "torch" or FRAMEWORK == "pytorch":
        prepare_torch()
        classify = torch_classify
        print("Using PyTorch")
    else:
        classify = mock_classify
        print("using MOCK framework")
    time.sleep(0.1)


if __name__ == '__main__':
    pr = cProfile.Profile()

    parse_arguments()
    prepare_framework()

    my_service = ThreadedServer(MyService, port=1200,
                                protocol_config={"allow_public_attrs": True})
    print("Waiting for a connection...")
    tpe.submit(run_job_batcher)
    tpe.submit(my_service.start)
