import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorrt as trt
from tensorrt.parsers import uffparser

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from random import randint # generate a random test case
from PIL import Image
import time #import system tools
import os

import uff

OUTPUT_NAMES = ["predictions/Softmax"]
PB_DIR =  "../tf_frozen/"
uff_model = uff.from_tensorflow_frozen_model(PB_DIR + "tf_inception.pb", OUTPUT_NAMES)
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

parser = uffparser.create_uff_parser()
parser.register_input("input_1", (3,299,299), 0)
parser.register_output(OUTPUT_NAMES[0])

engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)
parser.destroy()

img = np.random.randint(256, size=(3, 299, 299)) / 255.
img = img.astype(np.float32)

label = 123

runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

output = np.empty(1000, dtype = np.float32)

d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()


#transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
#execute model
context.enqueue(1, bindings, stream.handle, None)
#transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
#syncronize threads
stream.synchronize()

print("Test Case: " + str(label))
print ("Prediction: " + str(np.argmax(output)))

def infer():
    #transfer input data to device
    cuda.memcpy_htod_async(d_input, img, stream)
    #execute model
    context.enqueue(1, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    #syncronize threads
    stream.synchronize()

RUNS = 1000
total_run_time = 0

for i in range(1000):
    img = np.random.randint(256, size=(3, 299, 299)) / 255.
    img = img.astype(np.float32)
    start = time.time()
    infer()
    end = time.time()

    run_time = end-start
    total_run_time += run_time
    print("Test vs Pred: " + str(label) + " " + str(np.argmax(output)))
    print("Ran for: " + str(1000 * (end-start)) + " ms")
print("Done.")
print("Avg: " + str(1000 * (total_run_time / RUNS)) + " ms")
