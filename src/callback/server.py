import _thread as thread
from concurrent.futures import ThreadPoolExecutor

import rpyc
import time
from rpyc.utils.server import ThreadedServer

# import queue

tpe = ThreadPoolExecutor(max_workers=5)

items = []  # queue.Queue()


# class CallbackTest(object):
#     def __init__(self, callback, args):
#         self.callback = callback
#         self.args = args
#
#     def start(self):
#         thread.start_new_thread(self.callback, self.args)


class MyService(rpyc.Service):
    def __init__(self, conn):
        super().__init__(conn)
        print("MASTER INIT")

    def on_connect(self):
        """
        Invoked when the client connects.  Performs required initialization
        """
        print("Executive connected", thread.get_ident())

    def on_disconnect(self):
        """
        Invoked when the client disconnects.
        """
        print("Executive has disconnected...")

    def exposed_RemoteCallbackTest(self, i, callback):
        print("Got something.")
        items.append((i, callback))


def run_job_batcher():
    batch_size = 4
    while True:
        if len(items) >= batch_size:
            print("J", end="")
            inputs = [None] * batch_size
            callbacks = [None] * batch_size
            #
            for i in range(batch_size):
                inputs[i], callbacks[i] = items.pop(0)

            outputs = list(map(lambda x: x + 0, inputs))

            for i in range(batch_size):
                callbacks[i](outputs[i])


my_service = ThreadedServer(MyService, port=1200,
                            protocol_config={"allow_public_attrs": True})
print("Waiting for a connection...")
tpe.submit(run_job_batcher)
tpe.submit(my_service.start)
