from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from memory_profiler import profile
import argparse
import cProfile
import io as io2
import math
import pstats
import sys
import time
from datetime import datetime
import gc

import numpy as np
import torch

torch.backends.cudnn.benchmark = True

from six.moves import xrange
from torch.autograd import Variable
from utils import get_image_size, round_ms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchvision.datasets.fakedata import FakeData
FLAGS = None
g = None
options = None
prep_start = 0

def prepare_torch(model):
    print("Preparing Torch")
    global net

    model_file = "torch_frozen/torch_" + str(model) + ".out"

    assert torch.cuda.is_available()
    net = torch.load(model_file)
    net.cuda()
    print(net)


def torch_classify(imgs):
    res = net.forward(imgs) #.cuda(async=True)) #False)) #True)) #async=True)) #.cuda(async=True)) #async=True)) 
    torch.cuda.synchronize()
    res.cpu()


def time_run(info_string, imgs, fw, model):
    global run_inference, run_metadata, run_options, prep_start
    """Run the computation and print timing stats.

    Args:
      session: the TensorFlow session to run the computation under.
      target: the target Tensor that is passed to the session's run() function.
      info_string: a string summarizing this run, to be printed with the stats.

    Returns:
      None
    """
    my_dataset = FakeData(size=25000, transform=transforms.ToTensor())
    dataset_loader = torch.utils.data.DataLoader(my_dataset,
                                                 batch_size=FLAGS.batch_size,
                                                 shuffle=False,
                                                 num_workers=6, 
                                                 pin_memory=True) #False) #True)
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    total_process_duration = 0.0
    total_process_duration_squared = 0.0

    file_name = 'tf_traces/batch_' + str(fw) + "_" + str(model) + str(FLAGS.batch_size) + '.json'
    torch_chrome_file_name = 'torch_traces/batch_' + str(fw) + "_" + str(model) + str(FLAGS.batch_size) + '.json'
    dump_profiler_file_name = 'profiler_traces/' + str(fw) + "_" + str(model) + str(FLAGS.batch_size) + '.pstats'

    chrome = True
    pr = cProfile.Profile()

    output_trace = FLAGS.trace
    print("Output trace: " + str(output_trace))
    # FIX FOR 2.7

    if 'process_time' not in dir(time):
        time.process_time = time.clock  # Processor time

    
    start_time = time.time()
    for i, data in enumerate(dataset_loader): #i in xrange(FLAGS.num_batches + num_steps_burn_in):
        if i == FLAGS.num_batches + num_steps_burn_in:
            break
        gc.collect()
        
        batch = Variable(data[0], volatile=True, requires_grad=False) 
        batch = batch.cuda(async=True) #False)

        start_p_time = time.process_time()
        start_time = time.time()

        if output_trace:
            if not chrome and i >= num_steps_burn_in:
                pr.enable()

            if fw in {"torch", "pytorch"}:
                if chrome:
                    with torch.autograd.profiler.profile() as torch_prof:
                        run_inference(batch)
                    torch_prof.export_chrome_trace(torch_chrome_file_name)
                else:
                    pass

            if not chrome and i >= num_steps_burn_in:
                pr.disable()
        else:
            run_inference(batch)

        duration = time.time() - start_time
        p_duration = time.process_time() - start_p_time


        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.1f ms' %
                      (datetime.now(), i - num_steps_burn_in, 1000 * duration))
            total_duration += duration
            total_duration_squared += duration * duration
            total_process_duration += p_duration
            total_process_duration_squared += p_duration * p_duration

        if i == 0:
            print(datetime.now(), "Classified first batch, time since start (ms): ",
                  int(round_ms(time.time() - prep_start)))

    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)

    mean_process = total_process_duration / FLAGS.num_batches
    var_process = total_process_duration_squared / FLAGS.num_batches - mean_process * mean_process
    sd_process = math.sqrt(var_process)

    img_per_sec = FLAGS.batch_size * (1.0 / mn)

    info_str1 = ('%s: %s across %d steps (WALL TIME), %.4f +/- %.4f sec / batch, throughput (IPS) - %.3f %.4f\n' %
                 (datetime.now(), info_string, FLAGS.num_batches, mn, sd, img_per_sec, mn))
    info_str2 = ('%s: PROCESS_TIME: %.3f +/- %.3f sec / batch\n' %
                 (datetime.now(), mean_process, sd_process))
    info_str1 = info_str1 + info_str2
    print(info_str1, end="")
    with open("bench_logs/" + str(fw) + "log.out", "a") as myfile:
        myfile.write(info_str1)


def generate_dummy_images(fw, batch_size, image_size):
    if fw in {"torch", "pytorch", "trt", "tensorrt"}:
        images = np.random.randint(256, size=(batch_size, 3, image_size, image_size)) / 255.
    else:
        raise RuntimeError("Mock images not defined for framework: " + str(fw))

    print("Images shape", images.shape)
    return images.astype(np.float32)


def run_benchmark(fw, model):
    global g

    image_size = get_image_size(model)
    # Generate some dummy images.
    # images = "IMG"
    images = generate_dummy_images(fw, FLAGS.batch_size, image_size)

    info_string = str(fw) + ", " + str(model) + ", batch_size " + str(FLAGS.batch_size) + " |"
    time_run(info_string, images, fw, model)


# #@profile
def prepare_benchmark(fw, model):
    global run_inference

    print("Params: ", fw, "running", model)

    if fw in {"pytorch", "torch"}:
        prepare_torch(model)
        run_inference = torch_classify
    else:
        raise RuntimeError("No framework with this name")


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

    run_benchmark(framework, model)

    # s = io2.StringIO()
    # ps = pstats.Stats(pr).sort_stats("cumulative")
    # ps.print_stats()
    # prep_pstats = str(framework) + "_" + str(model) + "_prep.pstats"
    # ps.dump_stats(prep_pstats)


def get_argument_parser():
    global parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size','-b',
        type=int,
        default=1,
        help='Batch size.'
    )
    parser.add_argument(
        '--num_batches', '-n',
        type=int,
        default=20,
        help='Number of batches to run.'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default="torch",
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
    main(FLAGS)
