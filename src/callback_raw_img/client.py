import rpyc
import time
import glob
import argparse

# from concurrent.futures import ThreadPoolExecutor

# tpe = ThreadPoolExecutor(max_workers=4)


TOTAL_SENT = 500
TIME_BETWEEN_REQUESTS = 0.02

total_latency = 0.0
TOTAL_RECEIVED = 0


def callback_func_higher(i, start_time):
    def cb_func(x):
        global total_latency, count_latency, TOTAL_RECEIVED
        latency = time.time() - start_time
        print("GOT", i, x, latency)
        total_latency += latency
        TOTAL_RECEIVED += 1

    return cb_func


def report_stats():
    print("TOTAL LATENCY", total_latency)
    print("COUNT:", TOTAL_RECEIVED)
    print("AVG (ms):", 1000 * (total_latency / TOTAL_RECEIVED))


def parse_arguments():
    global TOTAL_SENT, TIME_BETWEEN_REQUESTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", help="num of total requests", type=int)
    parser.add_argument("--time", help="time between each request", type=float)
    args = parser.parse_args()
    if args.requests is not None:
        TOTAL_SENT = args.requests

    if args.time is not None:
        TIME_BETWEEN_REQUESTS = args.time
    print("Sending", TOTAL_SENT, "requests")
    print("Sleeping", TIME_BETWEEN_REQUESTS, "between each")
    time.sleep(0.1)


if __name__ == '__main__':
    parse_arguments()
    filenames = glob.glob("/Users/justas/PycharmProjects/ugproject/img/*.jpg")  # assuming gif

    host = "localhost"
    port = 1200
    conn = rpyc.connect(host, port)

    rpyc.BgServingThread.SLEEP_INTERVAL = 0
    rpyc.BgServingThread(conn)

    t1 = time.time()

    for i in range(TOTAL_SENT):
        t2 = time.time()
        if i % 100 == 0:
            print("SENT", i, t2 - t1)
        t1 = t2

        test = conn.root.RemoteCallbackTest(filenames[i], callback_func_higher(i, start_time=time.time()), )
        time.sleep(TIME_BETWEEN_REQUESTS)

    while TOTAL_RECEIVED < TOTAL_SENT:
        time.sleep(0.05)
    report_stats()
