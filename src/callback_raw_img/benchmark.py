from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time
import numpy as np
from torch.autograd import Variable
import torch
from tensorflow.python.client import timeline

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import torchvision

FLAGS = None


def prepare_torch(model):
    print("Preparing Torch")
    global net

    if model == "alexnet":
        from torch_models.torch_alexnet import alexnet
        net = alexnet(pretrained=True)  # works
    elif model == "vgg":
        from torch_models.torch_vgg import vgg16
        net = vgg16(pretrained=True)  # works
    elif model == "inception":
        from torch_models.torch_inception import inception_v3
        net = inception_v3(pretrained=True)  # 299 x 299 x 3
    elif model == "resnet":
        from torch_models.torch_resnet import resnet50
        net = resnet50(pretrained=True)  # works
    # torch.set_num_threads(4)

    print(net)


def torch_classify(imgs):
    transform = torchvision.transforms.ToTensor()
    imgs = list(map(transform, imgs))

    imgs = torch.stack(imgs)
    batch = Variable(imgs)

    try:
        net.forward(batch)
    except Exception as e:
        print(e)


def prepare_keras(model):
    print("Preparing Keras")
    import models_keras
    from keras import backend as K
    global net, graph

    if model == "alexnet":
        net = models_keras.create_model_alex()
    elif model == "vgg":
        net = models_keras.create_model_vgg16()
    elif model == "inception":
        net = models_keras.create_model_inception_v3()
    elif model == "resnet":
        net = models_keras.create_model_resnet50()

    graph = tf.get_default_graph()

    keras_session = K.get_session()

    print("Keras sesssion:", keras_session)

    # Easiest way to make the model build and compute the prediction function
    net.predict(np.random.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8))

    # net.build()
    # net.model._make_predict_function()


def keras_classify(imgs):
    global graph
    with graph.as_default():
        net.predict(imgs)


def prepare_tf(model):
    print("Preparing TensorFlow")
    global main_sess, images_placeholder, g, g_input, g_output

    g = load_frozen_tensorflow_graph(model)

    input_tensor_name = g.get_operations()[0].name + ":0"
    output_tensor_name = g.get_operations()[-1].name + ":0"

    with g.as_default():
        g_input = g.get_tensor_by_name(input_tensor_name)
        g_output = g.get_tensor_by_name(output_tensor_name)

        init_global = tf.global_variables_initializer()

    config = tf.ConfigProto()

    # Not relevant for non-GPU
    # config.gpu_options.allocator_type = 'BFC'

    main_sess = tf.Session(graph=g, config=config)
    main_sess.run(init_global)
    main_sess.run(g_output, feed_dict={g_input: (np.random.randint(256, size=(1, 224, 224, 3)))})


def load_frozen_tensorflow_graph(model):
    model_file = "tf_frozen/"

    if model == "alexnet":
        model_file += "tf_alex.pb"
    elif model == "vgg":
        model_file += "tf_vgg_frozen.pb"
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


def tf_classify(imgs, opts=None, run_md=None):
    # global g, g_input, g_output
    main_sess.run(g_output, options=opts, run_metadata=run_md, feed_dict={g_input: imgs})


def time_run(session, target, info_string, imgs, model):
    global run_inference
    """Run the computation and print timing stats.

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
    file_name = 'tf_traces/batch_' + str(model) + str(FLAGS.batch_size) + '.json'

    for i in xrange(FLAGS.num_batches + num_steps_burn_in):

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start_time = time.time()

        run_inference(imgs)  # , opts=options, run_md=run_metadata)

        # with torch.autograd.profiler.profile() as prof:
        #     torch_classify(imgs)
        # print(prof)

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


def run_benchmark(model):
    main_sess = None

    try:
        g is not None
    except:
        g = tf.get_default_graph()

    """Run the benchmark on AlexNet."""
    with g.as_default():
        if model in {"alexnet", "resnet", "vgg"}:
            image_size = 224
        elif model in {"inception"}:
            image_size = 299

        # Generate some dummy images.
        # Note that our padding definition is slightly different the cuda-convnet.
        # In order to force the model to start with the same activations sizes,
        # we add 3 to the image_size and employ VALID padding above.

        # Build an initialization operation.
        # init_local = tf.local_variables_initializer()
        # init = tf.global_variables_initializer()
        #
        # # Start running operations on the Graph.
        #
        # main_sess.run(init)
        # main_sess.run(init_local)

        # Run the forward benchmark.
        images = np.random.randint(256, size=(FLAGS.batch_size, image_size, image_size, 3)) / 255.
        print("Images shape", images.shape)
        time_run(main_sess, None, "Forward", images, model)


def prepare_benchmark():
    global run_inference
    model = FLAGS.model.lower()
    fw = FLAGS.framework.lower()
    print("Params: ", fw, "running", model)

    if fw in {"tensorflow", "tf"}:
        prepare_tf(model)
        run_inference = tf_classify
    elif fw in {"pytorch", "torch"}:
        prepare_torch(model)
        run_inference = torch_classify
    elif fw in {"keras"}:
        prepare_keras(model)
        run_inference = keras_classify
    else:
        raise RuntimeError("No framework with this name")


def main(_):
    prepare_benchmark()
    run_benchmark(FLAGS.model)


def get_argument_parser():
    global parser
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
        default=20,
        help='Number of batches to run.'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default="tensorflow",
        help='Framework to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="alexnet",
        help='ConvNet model to use'
    )
    return parser


if __name__ == '__main__':
    parser = get_argument_parser()

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
