import time
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf

q = tf.FIFOQueue(capacity=100, dtypes=tf.int32)


def run_producer():
    i = 0
    while True:
        time.sleep(0.3)
        i += 11
        with tf.Session() as sess:
            print("PRODUCE")
            # item = tf.Variable(i, dtype=tf.int32)
            q_enq = q.enqueue(tf.Variable(999, dtype=tf.int32))
            q_enq.run()
            print("size: ", sess.run(q.size()))


def run_inference(batch):
    print("INF before")
    with tf.Session() as sess:
        print(sess.run(batch))
        print("INF after")


def run_consumer():
    # latency_sum = 0
    # latency_count = 0
    while True:
        time.sleep(1)
        batch_img = q.dequeue()
        print("Batch", batch_img)
        run_inference(batch_img)
        # consumed = []
        # latency_count += 1
        # while q:
        #     msg, submit_time = q.pop()
        #     latency = time.time() - submit_time
        #     consumed.append(msg)
        #     latency_sum += latency
        # if latency_count % 5 == 0:
        #     print("mean latency: ", round(latency_sum / latency_count, 4))


# # q.pop(0)

# prod = produce()
# for x in prod:
#     print(x)
with ThreadPoolExecutor(max_workers=2) as e:
    e.submit(run_producer)
    # e.submit(run_consumer)

# prod = get_producer()
# for item in prod:
#     q.append(item)
#
# consumer()
