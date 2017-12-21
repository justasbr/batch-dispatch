#!/usr/bin/env python
from concurrent.futures import ThreadPoolExecutor

import pika
import queue
import time

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')

BATCH_SIZE = 1
items = queue.Queue()

tpe = ThreadPoolExecutor(max_workers=2)


def fib(n):
    return 0


def classify(inputs):
    return list(map(fib, inputs))


def process_and_return(batch_size):
    inputs = [None] * batch_size
    callbacks = [None] * batch_size

    for i in range(batch_size):
        inputs[i], callbacks[i] = items.get()
    outputs = classify(inputs)

    for i in range(batch_size):
        process_one_item(outputs[i], callbacks[i])


def process_one_item(output, other):
    ch, method, props = other
    print(ch, method, props)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(output))
    ch.basic_ack(delivery_tag=method.delivery_tag)

    # print(" [.] output = %s" % output)
    # cb(output)


def run_job_batcher():
    BATCH_TIMEOUT = 1

    batch_size = 1
    last_batch_time = None

    # TODO - now first batch always full
    while True:
        if items.qsize() >= batch_size:
            print("B" + str(batch_size), end=" ")
            last_batch_time = time.time()
            process_and_return(batch_size)
        elif items.qsize() and last_batch_time and (time.time() - last_batch_time) >= BATCH_TIMEOUT:
            items_to_process = items.qsize()
            print("B" + str(items_to_process), end="\n")
            last_batch_time = time.time()
            process_and_return(items_to_process)
        else:
            print("sleep")
            time.sleep(0.05)


def on_request(ch, method, props, body):
    # items.put(int(body), (ch, method, props))
    output = int(body)
    print("outputting")

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(output))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_size=0, prefetch_count=1000)
channel.basic_consume(on_request, queue='rpc_queue')

# tpe.submit(run_job_batcher)
# print("Job batcher")
tpe.submit(channel.start_consuming)
print(" [x] Awaiting RPC requests")
# channel.start_consuming()
