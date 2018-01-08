import _thread as thread
import rpyc
from concurrent.futures import ThreadPoolExecutor
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
MODEL = "alexnet"

tpe = ThreadPoolExecutor(max_workers=2)

image_queue = queue.Queue()
time_q = queue.Queue()

total_latency = 0.0
total_done = 0

total_classification_time = 0
classification_batches = 0

global classify, t1
graph = None


def prepare_keras(model):
    print("Preparing Keras")
    import models_keras
    global net, graph

    if model == "alexnet":
        net = models_keras.create_model_alex()
    elif model == "vgg":
        net = models_keras.create_model_vgg16()
    elif model == "inception":
        net = models_keras.create_model_inception_v3()
    elif model == "resnet":
        net = models_keras.create_model_resnet50()

    # net.save('keras_alex2.h5')
    # print("Saved")

    graph = tf.get_default_graph()

    # Easiest way to make the model build and compute the prediction function
    net.predict(np.random.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8))
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


def prepare_torch(model):
    print("Preparing Torch")
    global net, transform

    model_file = "torch_frozen/torch_" + str(model) + ".out"

    net = torch.load(model_file)
    # if model == "alexnet":
    #     from torch_models.torch_alexnet import alexnet
    #     net = alexnet(pretrained=True)  # works
    # elif model == "vgg":
    #     from torch_models.torch_vgg import vgg16
    #     net = vgg16(pretrained=True)  # works
    # elif model == "inception":
    #     from torch_models.torch_inception import inception_v3
    #     net = inception_v3(pretrained=True)  # 299 x 299 x 3
    # elif model == "resnet":
    #     from torch_models.torch_resnet import resnet50
    #     net = resnet50(pretrained=True)  # works

    print(net)

    transform = torchvision.transforms.ToTensor()


def torch_classify(imgs):
    # img_data = []
    # for img in imgs:
    #     img_data.append(Variable(transform(img)))

    img_data = list(map(lambda x: Variable(transform(x)), imgs))

    # print(img_data)
    # print("Batch", batch)
    batch = torch.stack(img_data)

    try:
        output = net.forward(batch)
    except Exception as e:
        print("Torch_classify failed, err: " + str(e))

    # print("Output", output)
    max_val, max_index = torch.max(output, 1)
    output = max_index.data.numpy()

    return output


def load_frozen_tensorflow_graph(model):
    model_file = "tf_frozen/"

    if model == "alexnet":
        # model_file += "tf_alexnetOP.pb"
        model_file += "tf_alex.pb"
    elif model == "vgg":
        model_file += "tf_vgg.pb"
    elif model == "inception":
        model_file += "tf_inception.pb"
    elif model == "resnet":
        model_file += "tf_resnet.pb"
    print("TF model file", model_file)

    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="main")
    return graph


def prepare_tf(model):
    print("Preparing TensorFlow")
    from alexnet_tf import alexnet_inference

    global fnames, enq_op, prob, main_sess, images_placeholder, g, g_input, g_output

    g = load_frozen_tensorflow_graph(model)

    for op in g.get_operations():
        print(op.name, op.type)

    input_tensor_name = g.get_operations()[0].name + ":0"
    output_tensor_name = g.get_operations()[-1].name + ":0"
    print(input_tensor_name, "<->", output_tensor_name)

    with g.as_default():
        g_input = g.get_tensor_by_name(input_tensor_name)
        g_output = g.get_tensor_by_name(output_tensor_name)

        # g_input = tf.placeholder(tf.uint8, shape=(None, 224, 224, 3))
        #
        # img_batch_float = tf.cast(g_input, tf.float32)
        # img_batch_float = tf.map_fn(tf.image.per_image_standardization, img_batch_float)
        # img_batch_float = tf.map_fn(lambda frame: tf.clip_by_value(frame, -1.0, 1.0), img_batch_float)
        # g_output, __ = alexnet_inference(img_batch_float)


        # image_batch = tf.stack(images)
        # print(image_batch) #/image_batch.dtype)

        # prob, __ = vgg_16(img_batch_float)
        # print(main_sess.run(prob))
        init_global = tf.global_variables_initializer()

    main_sess = tf.Session(graph=g)
    main_sess.run(init_global)
    main_sess.run(g_output, feed_dict={g_input: (np.random.randint(0, 200, size=(1, 224, 224, 3)))})


def tf_classify(imgs):
    global g, g_input, g_output
    # img_data = np.empty(shape=(len(imgs), 224, 224, 3))
    # for i, img in enumerate(imgs):
    #     print(img)
    # img_data[i] = img

    imgs = np.stack(imgs, axis=0)

    result = main_sess.run(g_output, feed_dict={g_input: imgs})
    # print(result)
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
        global total_latency, total_done
        print("Executive has disconnected...")
        print("Total latency (server side)", total_latency)
        print("Toral done", total_done)
        print("Avg latency server side (ms)", round_ms(total_latency / total_done))

    def exposed_RemoteCallbackTest(self, raw_img, callback):
        img = np.fromstring(raw_img, dtype=np.uint8)
        elems = int(img.shape[0])
        dim = int((elems / 3) ** 0.5)
        img = img.reshape(dim, dim, 3)
        image_queue.put((img, callback))
        t = time.time()
        # print("GOT", round_ms(t) % 10000)
        time_q.put(t)


def parse_arguments():
    global BATCH_SIZE, FRAMEWORK, BATCH_TIMEOUT, MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="batch_size", type=int)
    parser.add_argument("-t", "--timeout", help="timeout for batch", type=float)
    parser.add_argument("-f", "--framework", help="framework")
    parser.add_argument("-m", "--model", help="convnet model")

    args = parser.parse_args()
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size

    if args.framework is not None:
        FRAMEWORK = args.framework

    if args.timeout is not None:
        BATCH_TIMEOUT = args.timeout

    if args.model is not None:
        if args.model.startswith("vgg"):
            MODEL = "vgg"
        elif args.model.startswith("incep"):
            MODEL = "inception"
        elif args.model.startswith("res"):
            MODEL = "resnet"
        else:
            MODEL = "alexnet"

    print("Model:", MODEL)
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
    try:
        outputs = classify(inputs)
    except Exception as e:
        print("Classification error:", e)
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
        perform_function(callbacks[i], outputs[i])

        curr_latency = time.time() - time_q.get()
        total_latency += curr_latency
        total_done += 1


def perform_function(f, arg):
    tpe.submit(f, arg)


def prepare_framework():
    global classify
    if FRAMEWORK == "tf" or FRAMEWORK == "tensorflow":
        prepare_tf(MODEL)
        classify = tf_classify
        print("Using TensorFlow")
    elif FRAMEWORK == "torch" or FRAMEWORK == "pytorch":
        prepare_torch(MODEL)
        classify = torch_classify
        print("Using PyTorch")
    elif FRAMEWORK == "keras":
        prepare_keras(MODEL)
        classify = keras_classify
        print("Using Keras")
    else:
        classify = mock_classify
        print("using MOCK framework")


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
