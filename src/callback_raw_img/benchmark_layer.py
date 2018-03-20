from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pstats
from io import StringIO
import cProfile
# from memory_profiler import profile
import argparse
import math
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import torch
from keras import layers as keras_layers
from keras.models import Sequential
from six.moves import xrange
from tensorflow.python.client import timeline
from torch.autograd import Variable
from utils import get_image_size

FLAGS = None
g = None
run_metadata = tf.RunMetadata()
options = None
prep_start = 0


def prepare_torch():
    print("Preparing Torch")
    global net
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 48, (11, 11), stride=4, padding=0),
        torch.nn.Conv2d(48, 24, (11, 11), stride=4, padding=0),
    )

    if torch.cuda.is_available():
        net.cuda()
    print(net)


def torch_classify(imgs):
    batch = Variable(torch.from_numpy(imgs), requires_grad=False)
    if torch.cuda.is_available():
        r = net.forward(batch.cuda())
        r.cpu()
    else:
        r = net.forward(batch)
    return r.data.numpy() #numpy()


def prepare_keras():
    print("Preparing Keras")
    global net
    net = Sequential()
    net.add(keras_layers.Conv2D(48, (11, 11), strides=(4, 4), padding='valid', input_shape=(224, 224, 3)))


def keras_classify(imgs):
    return net.predict(imgs)
     

def prepare_tf():
    print("Preparing TensorFlow")
    global main_sess, g, g_input, g_output

    g_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    filter1 = tf.get_variable('weights1', [11, 11, 3, 48],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
                              dtype=tf.float32)

    filter2 = tf.get_variable('weights2', [11,11, 48,24],
                              initializer=tf.truncated_normal_initializer(stddev=3e-2, dtype=tf.float32),
                              dtype=tf.float32)

    g_output = tf.nn.conv2d(g_input, filter=filter1, strides=(1, 4, 4, 1), padding="VALID")
    g_output = tf.nn.conv2d(g_output, filter=filter2, strides=(1, 4, 4, 1), padding="VALID")

    main_sess = tf.Session(config=tf.ConfigProto())

    init_global = tf.global_variables_initializer()
    main_sess.run(init_global)


def tf_classify(imgs, options=None, run_metadata=None):
    return main_sess.run(g_output,options=options,run_metadata=run_metadata, feed_dict={g_input: imgs})
    

def time_run(info_string, imgs_gen):
    global run_inference, run_metadata, prep_start
    
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if True else None
    #run_metadata = tf.RunMetadata()
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
    for i in xrange(FLAGS.num_batches + num_steps_burn_in):
        images = next(imgs_gen)
        start_p_time = time.process_time()
        start_time = time.time()
        result = run_inference(images)
        #if FLAGS.framework == "tf":
        #    result = run_inference(images, options=run_options, run_metadata=run_metadata)
        #elif FLAGS.framework == "torch":# print(result[0][0][0])
        #    with torch.autograd.profiler.profile() as torch_prof:
        #        result = run_inference(images)
        #    torch_prof.export_chrome_trace("layers_torch.json")
        #else:
        #    result = run_inference(images)
        duration = time.time() - start_time
        p_duration = time.process_time() - start_p_time
        
        if i == 0:
            print(result.shape)

        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
            total_process_duration += p_duration
            total_process_duration_squared += p_duration * p_duration
    #if FLAGS.framework == "tf":
    #    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    #    chrome_trace = fetched_timeline.generate_chrome_trace_format(show_dataflow=True, show_memory=False)
    #
    #   with open("layers.json", 'w') as f:
    #        f.write(chrome_trace)

    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)

    mean_process = total_process_duration / FLAGS.num_batches
    var_process = total_process_duration_squared / FLAGS.num_batches - mean_process * mean_process
    sd_process = math.sqrt(var_process)

    print('%s: %s across %d steps (WALL TIME), %.4f +/- %.4f sec / batch' %
          (datetime.now(), info_string, FLAGS.num_batches, mn, sd))
    print('%s: PROCESS_TIME: %.3f +/- %.3f sec / batch' %
          (datetime.now(), mean_process, sd_process))

def generate_dummy_images(fw, batch_size, image_size):
    while True:
        if fw in {"torch", "pytorch"}:
            images = np.random.randint(256, size=(batch_size, 3, image_size, image_size))
        elif fw in {"tensorflow", "tf", "keras"}:
            images = np.random.randint(256, size=(batch_size, image_size, image_size, 3))
        else:
            raise RuntimeError("Mock images not defined for framework: " + str(fw))

        images = images / 255.0

        # print("Images shape", images.shape)
        yield images.astype(np.float32)


def run_benchmark(fw):
    global g
    if fw in {"keras"}:
        g = g or tf.get_default_graph()

    image_size = get_image_size("alexnet")
    # Generate some dummy images.
    images = generate_dummy_images(fw, FLAGS.batch_size, image_size)

    info_string = str(fw) + ", batch_size " + str(FLAGS.batch_size) + " |"
    time_run(info_string, images)


def prepare_benchmark(fw):
    global run_inference

    print("Params: ", fw)

    if fw in {"tensorflow", "tf"}:
        prepare_tf()
        run_inference = tf_classify
    elif fw in {"pytorch", "torch"}:
        prepare_torch()
        run_inference = torch_classify
    elif fw in {"keras"}:
        prepare_keras()
        run_inference = keras_classify
    else:
        raise RuntimeError("No framework with this name")


def main(_):
    framework = FLAGS.framework.lower()

    prepare_benchmark(framework)
    run_benchmark(framework)


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
    return parser


if __name__ == '__main__':
    parser = get_argument_parser()
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
    # main()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
