from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cProfile
import io as io2
import math
import pstats
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import torch
from keras import backend as K
from six.moves import xrange
from tensorflow.python.client import timeline
from torch.autograd import Variable
from utils import get_image_size, round_ms

FLAGS = None
g = None
run_metadata = tf.RunMetadata()
options = None
prep_start = 0


def prepare_torch(model):
    print("Preparing Torch")
    global net

    model_file = "torch_frozen/torch_" + str(model) + ".out"

    net = torch.load(model_file)
    if torch.cuda.is_available():
        net.cuda()
    # torch.set_num_threads(4)
    print(net)


def torch_classify(imgs, opts=None, run_md=None):
    batch = Variable(torch.from_numpy(imgs), requires_grad=False)
    if torch.cuda.is_available():
        r = net.forward(batch.cuda())
        r.cpu()
    else:
        net.forward(batch)


# #@profile
def prepare_keras(model):
    print("Preparing Keras")
    import models_keras
    global net, graph

    net = models_keras.create_model(model)
    net.summary()

    graph = tf.get_default_graph()
    keras_session = K.get_session()

    # if FLAGS.trace:
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # net.compile(loss='MSE', optimizer='Adam', options=run_options, run_metadata=run_metadata)

    # old_run = keras_session.run
    # print("old_run", old_run)
    # print("K.sess 1", keras_session)
    #
    # def new_run(*args, **kwargs):
    #     # print("X", self)
    #     print("NewRunning", *args)
    #     old_run(*args, **kwargs)

    # keras_session.run = new_run


# #@profile
def keras_classify(imgs, opts=None, run_md=None):
    global graph
    with graph.as_default():
        net.predict(imgs)


def prepare_tf(model):
    print("Preparing TensorFlow")
    global main_sess, images_placeholder, g, g_input, g_output

    g = load_frozen_tensorflow_graph(model)

    input_tensor_name = g.get_operations()[0].name + ":0"
    if model in {"alexnet"}:
        output_tensor_name = "dense_3/BiasAdd:0"
    else:
        output_tensor_name = g.get_operations()[-1].name + ":0"

    print("I:", input_tensor_name)
    print("O:", output_tensor_name)

    with g.as_default():
        g_input = g.get_tensor_by_name(input_tensor_name)
        g_output = g.get_tensor_by_name(output_tensor_name)

        init_global = tf.global_variables_initializer()

    config = tf.ConfigProto()

    # Not relevant for non-GPU
    # config.gpu_options.allocator_type = 'BFC'

    main_sess = tf.Session(graph=g, config=config)
    main_sess.run(init_global)


def load_frozen_tensorflow_graph(model):
    global main_sess, g
    # saver = tf.train.import_meta_graph('./tmp/model.ckpt-55695.meta')
    # saver.restore(session, './tmp/model.ckpt-55695')

    model_file = "./tf_frozen/" if model in {"alexnet"} else "tf_frozen/"

    if model == "alexnet":
        # model_file += "tf_alexnetOP.pb"
        model_file += "tf_alexnet.ckpt"

        with tf.Graph().as_default() as g, tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_file + ".meta")
            saver.restore(sess, model_file)
        print("TF model file", model_file)
    else:
        if model == "vgg":
            model_file += "tf_vgg.pb"
        elif model == "inception":
            model_file += "tf_inception.pb"
        elif model == "resnet":
            model_file += "tf_resnet.pb"
        print("TF model file", model_file)

        with tf.gfile.GFile(model_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as g:
            tf.import_graph_def(graph_def, name="main")
    return g


# #@profile
def tf_classify(imgs, opts=None, run_md=None):
    main_sess.run(g_output, options=opts, run_metadata=run_md, feed_dict={g_input: imgs})


def time_run(info_string, imgs, fw, model):
    global run_inference, run_metadata, prep_start
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

    total_process_duration = 0.0
    total_process_duration_squared = 0.0

    file_name = 'tf_traces/batch_' + str(fw) + "_" + str(model) + str(FLAGS.batch_size) + '.json'
    torch_chrome_file_name = 'torch_traces/batch_' + str(fw) + "_" + str(model) + str(FLAGS.batch_size) + '.json'
    dump_profiler_file_name = 'profiler_traces_NEW/' + str(fw) + "_" + str(model) + str(FLAGS.batch_size) + '.pstats'

    pr = cProfile.Profile()

    output_trace = FLAGS.trace
    print("Output trace: " + str(output_trace))

    for i in xrange(FLAGS.num_batches + num_steps_burn_in):

        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if output_trace else None
        # run_metadata = tf.RunMetadata()

        start_p_time = time.process_time()
        start_time = time.time()

        if output_trace:
            if i >= num_steps_burn_in:
                pr.enable()

            if fw in {"torch", "pytorch"}:
                # with torch.autograd.profiler.profile() as torch_prof:
                run_inference(imgs)
                # print(torch_prof.key_averages())
                # torch_prof.export_chrome_trace(torch_chrome_file_name)

            else:
                # run_inference(imgs, opts=options, run_md=run_metadata)
                run_inference(imgs)  # , opts=options, run_md=run_metadata)

            if i >= num_steps_burn_in:
                pr.disable()
        else:
            run_inference(imgs)

        duration = time.time() - start_time
        p_duration = time.process_time() - start_p_time

        if output_trace:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format(show_dataflow=True, show_memory=False)

            with open(file_name, 'w') as f:
                f.write(chrome_trace)

        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
            total_process_duration += p_duration
            total_process_duration_squared += p_duration * p_duration

        if i == 0:
            print(datetime.now(), "Classified first batch, time since start (ms): ",
                  int(round_ms(time.time() - prep_start)))

    if output_trace:
        # s = io2.StringIO()
        ps = pstats.Stats(pr).sort_stats("cumulative")
        # ps.print_stats()
        ps.dump_stats(dump_profiler_file_name)

    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)

    mean_process = total_process_duration / FLAGS.num_batches
    var_process = total_process_duration_squared / FLAGS.num_batches - mean_process * mean_process
    sd_process = math.sqrt(var_process)

    print('%s: %s across %d steps (WALL TIME), %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, FLAGS.num_batches, mn, sd))
    print('%s: PROCESS_TIME: %.3f +/- %.3f sec / batch' %
          (datetime.now(), mean_process, sd_process))


def generate_dummy_images(fw, batch_size, image_size):
    if fw in {"torch", "pytorch"}:
        images = np.random.randint(256, size=(batch_size, 3, image_size, image_size)) / 255.
    # elif True:
    #     images = np.random.randint(256, size=(batch_size, 3, image_size, image_size)) / 255.  # tmp
    elif fw in {"tensorflow", "tf", "keras"}:
        images = np.random.randint(256, size=(batch_size, image_size, image_size, 3)) / 255.
    else:
        raise RuntimeError("Mock images not defined for framework: " + str(fw))

    print("Images shape", images.shape)
    return images.astype(np.float32)


def run_benchmark(fw, model):
    global g
    if fw in {"keras"}:
        g = g or tf.get_default_graph()

    image_size = get_image_size(model)
    # Generate some dummy images.
    images = generate_dummy_images(fw, FLAGS.batch_size, image_size)

    info_string = str(fw) + ", " + str(model) + ", batch_size " + str(FLAGS.batch_size) + " |"
    time_run(info_string, images, fw, model)


# #@profile
def prepare_benchmark(fw, model):
    global run_inference

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


# @profile
def main(_):
    pr = cProfile.Profile()

    global prep_start
    model = FLAGS.model.lower()
    framework = FLAGS.framework.lower()

    prep_start = time.time()
    # pr.enable()

    prepare_benchmark(framework, model)

    # pr.disable()


    # prep_end = time.time()

    # print("PREP_TIME (ms)", int(round_ms(prep_end - prep_start)))

    # pr.enable()
    run_benchmark(framework, model)
    # pr.disable()

    # s = io2.StringIO()
    # ps = pstats.Stats(pr).sort_stats("cumulative")
    # ps.print_stats()
    # print("AYO")

    # ps.print_stats()
    # print(ps)
    # ps.print_stats(10)
    # prep_pstats = str(framework) + "_" + str(model) + "_prep.pstats"
    # print(ps)
    # ps.dump_stats(prep_pstats)


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
    parser.add_argument(
        '--trace',
        type=bool,
        default=False,
        help='trace'
    )
    return parser


if __name__ == '__main__':
    parser = get_argument_parser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
