import _thread as thread
from concurrent.futures import ThreadPoolExecutor

import rpyc
from rpyc.utils.server import ThreadedServer

import queue
import time

import torch
import torchvision
from torch.autograd import Variable
from skimage import io
import argparse
import random
import os
from utils import round_ms

import cProfile, pstats, io as io2

import tensorflow as tf
from numpy import prod
import numpy as np

BATCH_SIZE = 4
BATCH_TIMEOUT = 0.1
FRAMEWORK = ""

tpe = ThreadPoolExecutor(max_workers=4)

image_queue = queue.Queue()
time_q = queue.Queue()

total_latency = 0.0
total_done = 0

total_classification_time = 0
classification_batches = 0

global classify, t1
graph = None


def prepare_torch():
    from alexnet_torch import alexnet
    global net, transform
    net = alexnet(pretrained=True)
    transform = torchvision.transforms.ToTensor()


def prepare_keras():
    import alexnet_keras
    print("Preparing keras")
    global net, graph
    net = alexnet_keras.create_model_alex()
    graph = tf.get_default_graph()

    # Easiest way to make the model build and compute the prediction function
    net.predict(np.random.randint(0, 200, size=(1, 224, 224, 3)))
    # net.build()
    # net.model._make_predict_function()


def keras_classify(imgs):
    global graph
    with graph.as_default():
        imgs = np.stack(imgs, axis=0)
        # try:
        res = net.predict(imgs)
        # except Exception as e:
        #     print(e)
        # print(res)
        output = res.argmax(axis=1)
        # print(output)
        return output


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


def prepare_tf():
    from alexnet_tf import alexnet_inference
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
    main_sess = tf.Session(graph=g)
    main_sess.run(init_global)


def tf_classify(imgs):
    # img_data = np.empty(shape=(len(imgs), 224, 224, 3))
    # for i, img in enumerate(imgs):
    #     print(img)
    # img_data[i] = img

    imgs = np.stack(imgs, axis=0)
    result = main_sess.run(prob, feed_dict={images_placeholder: imgs})

    return result.argmax(axis=1)


def mock_classify(raw_images):
    def get_random_num(raw_img):
        return random.randint(0, 1000)

    return list(map(get_random_num, raw_images))


class MyService(rpyc.Service):
    def __init__(self, conn):
        super().__init__(conn)
        print("MASTER INIT")

    def on_connect(self):
        print("Executive connected", thread.get_ident())

    def on_disconnect(self):
        print("Executive has disconnected...")

    def exposed_RemoteCallbackTest(self, raw_img, callback):
        img = np.fromstring(raw_img, dtype=np.uint8).reshape(224, 224, 3)
        image_queue.put((img, callback))
        time_q.put(time.time())


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
    global total_classification_time, classification_batches

    inputs = [None] * batch_size
    callbacks = [None] * batch_size
    for i in range(batch_size):
        inputs[i], callbacks[i] = image_queue.get()

    classify_start = time.time()

    # pr.enable()
    # THE CLASSIFICATION
    outputs = classify(inputs)
    # pr.disable()

    classify_time = time.time() - classify_start
    total_classification_time += classify_time
    classification_batches += 1

    print("batch=" + str(batch_size) +
          ", took (ms) " + str(round_ms(classify_time)) +
          ", avg (ms) " + str(round_ms(total_classification_time / classification_batches)))

    # s = io2.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    # ps.print_stats()
    # print(s.getvalue())

    global total_latency, total_done
    for i in range(batch_size):
        curr_latency = time.time() - time_q.get()

        total_latency += curr_latency
        total_done += 1

        perform_function(callbacks[i], outputs[i])


def perform_function(f, arg):
    tpe.submit(f, arg)


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
    elif FRAMEWORK == "keras":
        prepare_keras()
        classify = keras_classify
        print("Using Keras")
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
    my_service.start()
    # tpe.submit(my_service.start)
