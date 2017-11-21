import rpyc
import glob
import time
import argparse
from skimage import io
import random
import numpy as np

from utils import round_ms
from concurrent.futures import ThreadPoolExecutor
import cProfile, pstats, io as io2


tpe = ThreadPoolExecutor(max_workers=None)
#print(tpe)
tpe_out = ThreadPoolExecutor(max_workers=15)

TOTAL_SENT = 500
TIME_BETWEEN_REQUESTS = 0.02

total_latency = 0.0
TOTAL_RECEIVED = 0
IMG_SIZE = None
moving_latency_average = 0.0
first_packet_time = None
last_packet_time = None
ALPHA = 0.925  # moving average


def callback_func_higher(i, start_time):
    def cb_func(x):
        # print("Got answer", time.time())
        global total_latency, count_latency, TOTAL_RECEIVED, ALPHA, moving_latency_average
        latency = time.time() - start_time

        moving_latency_average = ALPHA * moving_latency_average + (1 - ALPHA) * latency
        if not i % 100:
            print("GOT " + str(i) + " c: " + str(x) +
                  " LAT: " + str(round_ms(latency)) +
                  " Moving LAT avg:" + str(round_ms(moving_latency_average)))
        total_latency += latency
        TOTAL_RECEIVED += 1

    return cb_func


def report_stats():
    global last_packet_time, first_packet_time
    print("COUNT:", TOTAL_RECEIVED)
    time_to_send_all = last_packet_time - first_packet_time
    time_to_send_one = time_to_send_all / TOTAL_RECEIVED
    packets_sent_per_second = 1 / time_to_send_one
    print("TOTAL time taken to send (ms):", round_ms(time_to_send_all))
    print("Packets sent per sec", packets_sent_per_second)
    print("AVG latency (ms):", round_ms(total_latency / TOTAL_RECEIVED))


def parse_arguments():
    global TOTAL_SENT, TIME_BETWEEN_REQUESTS, IMG_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", help="num of total requests", type=int)
    parser.add_argument("--time", help="time between each request", type=float)
    parser.add_argument("--size", help="size of img (pix)", type=int)
    args = parser.parse_args()
    if args.requests is not None:
        TOTAL_SENT = args.requests

    if args.time is not None:
        TIME_BETWEEN_REQUESTS = args.time

    if args.size is not None:
        IMG_SIZE = args.size

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
    #tpe.submit(rpyc.BgServingThread, conn)

    filenames = glob.glob("/Users/justas/PycharmProjects/ugproject/img/*.jpg")  # assuming gif
    random.shuffle(filenames)

    t1 = time.time()
    first_packet_time = time.time()
    for i in range(TOTAL_SENT):
        t2 = time.time()
        #if not ((i + 1) % 50):
        #    print("SENT", i, t2 - t1)
        t1 = t2

        if IMG_SIZE is not None:
            img_numpy = np.random.randint(256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            # print(img_numpy.shape)
        else:
            file_name = filenames[i]
            img_numpy = io.imread(file_name)
            # print(img_numpy)

        raw_data = img_numpy.tobytes()

        tpe_out.submit(send_img, conn, i, raw_data)
        time.sleep(TIME_BETWEEN_REQUESTS)

    while TOTAL_RECEIVED < TOTAL_SENT:
        time.sleep(0.01)
    report_stats()


def send_img(conn, i, raw_data):
    conn.root.RemoteCallbackTest(raw_data, callback_func_higher(i, start_time=time.time()), )
    if not i % 50:
        print("SENT\t" +  str(i) + " " + str(time.time()))
    # print("Sent img", time.time())
    # pr.enable()

    if i == 0:
        global first_packet_time
        first_packet_time = time.time()
    if i == (TOTAL_SENT - 1):
        global last_packet_time
        last_packet_time = time.time()

        # pr.disable()

    # s = io2.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


if __name__ == '__main__':
    pr = cProfile.Profile()
    main()
