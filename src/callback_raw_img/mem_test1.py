# import numpy as np
from memory_profiler import profile
# import os
# import gc

# @profile
# def mem_test():
#     image_size = 400
#     images = []
#     for i in range(1000):
#         x = np.random.randint(256, size=(1, 3, image_size, image_size))
#         images.append(x)
#         with open('test_file.out', 'wb') as f:
#             f.write(x.tobytes())
#             f.close()
#
#     for i in range(10):
#         images.pop()
#
#     np.mean(images[-1])


# @profile
def mem_test2():
    a = [0] * (10 ** 6)
    b = a*2
    c = b*2
    return a


mem_test2()
