import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorrt as trt
from tensorrt.parsers import uffparser

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
# from PIL import Image
import time #import system tools
import os

import uff

PB_DIR = "./tf_frozen/"

OUTPUT_NAMES = ["activation_8/Softmax"]

def create_infer(BATCH_SIZE):

    uff_model = uff.from_tensorflow_frozen_model(PB_DIR + "tf_alex.pb", OUTPUT_NAMES)
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    parser = uffparser.create_uff_parser()
    parser.register_input("conv2d_1_input", (3,224,224),0)
    parser.register_output(OUTPUT_NAMES[0])

    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, BATCH_SIZE, 0) # 1 << 20

    img = np.random.randint(256, size=(BATCH_SIZE, 3, 224, 224)) / 255.
    img = img.astype(np.float32)

    label = 123

    runtime = trt.infer.create_infer_runtime(G_LOGGER)
    context = engine.create_execution_context()

    output = np.empty((BATCH_SIZE, 1000), dtype = np.float32)

    d_input = cuda.mem_alloc(BATCH_SIZE * img.size * img.dtype.itemsize)
    d_output = cuda.mem_alloc(BATCH_SIZE * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    def infer(img):
        #transfer input data to device
        cuda.memcpy_htod_async(d_input, img, stream)
    #execute model
        context.enqueue(BATCH_SIZE, bindings, stream.handle, None)
    #transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
    #syncronize threads
        stream.synchronize()
        return output
    return infer
