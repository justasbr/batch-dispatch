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

OUTPUT_NAMES = ["fc1000/Softmax"]

BATCH_SIZE = 2
PB_DIR = "../tf_frozen/"
uff_model = uff.from_tensorflow_frozen_model(PB_DIR + "tf_resnet.pb", OUTPUT_NAMES)
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

parser = uffparser.create_uff_parser()
parser.register_input("input_1", (3,224,224), 0)
parser.register_output(OUTPUT_NAMES[0])

engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, BATCH_SIZE, 0) #1 << 20) #, datatype=trt.infer.DataType.HALF)
parser.destroy()

img = np.random.randint(256, size=(3, 224, 224)) / 255.
img = img.astype(np.float32)

label = 123

runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

output = np.empty((BATCH_SIZE,1000), dtype = np.float32)

d_input = cuda.mem_alloc(BATCH_SIZE * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(BATCH_SIZE * output.size * output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

def infer(img_input):
    global output
    #transfer input data to device
    cuda.memcpy_htod_async(d_input, img_input, stream)
    #execute model
    context.enqueue(BATCH_SIZE, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    #syncronize threads
    stream.synchronize()

total_run_time = 0.0
RUNS = 1000

for i in range(RUNS):
    img = np.random.randint(256, size=(BATCH_SIZE, 3, 224, 224)) / 255.
    img = img.astype(np.float32)
    start = time.time()
    infer(img)
    end = time.time() 
    print("Ran for: " + str(1000 * (end-start)) + " ms\t\t" + "test v prod " + str(label) + " " + str(np.argmax(output, axis=1)))
    total_run_time += (end-start)
print("Done.")
print("AVG: " + str(1000 * (total_run_time / RUNS))) 
