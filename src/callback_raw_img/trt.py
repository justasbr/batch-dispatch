import tensorrt as trt
from tensorrt.parsers import uffparser

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
# from PIL import Image
import time  # import system tools
import os
import uff

PB_DIR = "./tf_frozen/"

config = {
    "alexnet": {"file": PB_DIR + "tf_alex.pb",
                "input_layer": "conv2d_1_input",
                "output_layer": ["activation_8/Softmax"],
                "size": 224},

    "vgg": {"file": PB_DIR + "tf_vgg.pb",
            "input_layer": "input_1",
            "output_layer": ["predictions/Softmax"],
            "size": 224},

    "resnet": {"file": PB_DIR + "tf_resnet.pb",
               "input_layer": "input_1",
               "output_layer": ["fc1000/Softmax"],
               "size": 224},

    "inception": {"file": PB_DIR + "tf_inception.pb",
                  "input_layer": "input_1",
                  "output_layer": ["predictions/Softmax"],
                  "size": 299}
}


def get_inference_handle(model, batch_size):
    print("Batch size", batch_size)
    if model not in {"alexnet", "vgg", "resnet", "inception"}:
        raise Exception("TRT did not have model:" + model)
    else:
        return create_infer(model, batch_size)


def create_infer(model, batch_size):
    SETTINGS = config[model]
    file = SETTINGS["file"]
    output_names = SETTINGS["output_layer"]
    input_name = SETTINGS["input_layer"]
    img_size = SETTINGS["size"]

    uff_model = uff.from_tensorflow_frozen_model(file, output_names)
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    parser = uffparser.create_uff_parser()
    parser.register_input(input_name, (3, img_size, img_size), 0)
    parser.register_output(output_names[0])

    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, batch_size, 0)  # 1 << 20

    imgs_size = batch_size * 3 * img_size * img_size
    float32_size = np.dtype('float32').itemsize

    context = engine.create_execution_context()

    outputs = np.empty((batch_size, 1000), dtype=np.float32)

    d_input = cuda.mem_alloc(imgs_size * float32_size)
    d_output = cuda.mem_alloc(outputs.size * float32_size)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    def infer(imgs):
        cuda.memcpy_htod_async(d_input, imgs, stream)
        context.enqueue(batch_size, bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(outputs, d_output, stream)
        stream.synchronize()
        return outputs

    return infer
