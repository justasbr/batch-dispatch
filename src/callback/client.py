import rpyc
import time


# from concurrent.futures import ThreadPoolExecutor

# tpe = ThreadPoolExecutor(max_workers=4)


def callback_func_higher(start_time):
    def cb_func(x):
        print("GOT", x, time.time() - start_time)

    return cb_func
    # return callback_func(x)


def callback_func(x):
    print(x, time.time())


def go():
    host = "localhost"
    port = 1200
    conn = rpyc.connect(host, port)

    rpyc.BgServingThread.SLEEP_INTERVAL = 0
    rpyc.BgServingThread(conn)

    t1 = time.time()
    for i in range(1000):
        t2 = time.time()
        print("SENT", i, t2 - t1)
        t1 = t2
        # test = conn.root.RemoteCallbackTest(i, callback_func) #(start_time=time.time()), )

        test = conn.root.RemoteCallbackTest(i, callback_func_higher(start_time=time.time()), )
        # test = conn.root.RemoteCallbackTest(i, callback_func_higher(start_time=time.time()))
        # time.sleep(0.1)


go()
