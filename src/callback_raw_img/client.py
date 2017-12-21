import rpyc
import glob
import time
import argparse
from skimage import io
import random

from utils import round_ms
from concurrent.futures import ThreadPoolExecutor
import cProfile, pstats, io as io2

# from gprof import GProfiler

tpe = ThreadPoolExecutor(max_workers=2)

TOTAL_SENT = 500
TIME_BETWEEN_REQUESTS = 0.02

total_latency = 0.0
TOTAL_RECEIVED = 0
moving_latency_average = 0.0
first_packet_time = None
last_packet_time = None
ALPHA = 0.95  # moving average


def callback_func_higher(i, start_time):
    def cb_func(x):
        # print("Got answer", time.time())
        global total_latency, count_latency, TOTAL_RECEIVED, ALPHA, moving_latency_average
        latency = time.time() - start_time

        moving_latency_average = ALPHA * moving_latency_average + (1 - ALPHA) * latency
        print("GOT " + str(i) + " c: " + str(x) +
              " LAT: " + str(round_ms(latency)) +
              " Moving LAT avg:" + str(round_ms(moving_latency_average)))
        total_latency += latency
        TOTAL_RECEIVED += 1

    return cb_func


def report_stats():
    print("COUNT:", TOTAL_RECEIVED)
    time_to_send_all = last_packet_time - first_packet_time
    time_to_send_one = time_to_send_all / TOTAL_RECEIVED
    packets_sent_per_second = 1 / time_to_send_one
    print("TOTAL time taken to send (ms):", round_ms(time_to_send_all))
    print("Packets sent per sec", packets_sent_per_second)
    print("AVG latency (ms):", round_ms(total_latency / TOTAL_RECEIVED))


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


def main():
    global first_packet_time, last_packet_time
    parse_arguments()
    host = "localhost"
    port = 1200
    conn = rpyc.connect(host, port)
    rpyc.BgServingThread.SLEEP_INTERVAL = 0
    rpyc.BgServingThread(conn)
    # tpe.submit(rpyc.BgServingThread, conn)
    filenames = glob.glob("/Users/justas/PycharmProjects/ugproject/img/*.jpg")  # assuming gif
    random.shuffle(filenames)

    t1 = time.time()
    first_packet_time = time.time()
    for i in range(TOTAL_SENT):
        t2 = time.time()
        if not ((i + 1) % 50):
            print("SENT", i, t2 - t1)
        t1 = t2

        file_name = filenames[i]
        raw_data = io.imread(file_name).tobytes()

        tpe.submit(send_img, conn, i, raw_data)
        time.sleep(TIME_BETWEEN_REQUESTS)
    last_packet_time = time.time()

    while TOTAL_RECEIVED < TOTAL_SENT:
        time.sleep(0.01)
    report_stats()


def send_img(conn, i, raw_data):
    # print("Sent img", time.time())
    # pr.enable()
    conn.root.RemoteCallbackTest(raw_data, callback_func_higher(i, start_time=time.time()), )
    # pr.disable()

    # s = io2.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


if __name__ == '__main__':
    pr = cProfile.Profile()
    main()
