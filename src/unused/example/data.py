import threading
import time
import tensorflow as tf
import numpy as np


def load_data():
    items = [
        "/Users/justas/PycharmProjects/ugproject/img/img1.jpg",
        "/Users/justas/PycharmProjects/ugproject/img/img2.jpg",
        "/Users/justas/PycharmProjects/ugproject/img/img46.jpg",
        "/Users/justas/PycharmProjects/ugproject/img/img47.jpg",
        "/Users/justas/PycharmProjects/ugproject/img/img48.jpg"]
    for i in range(10000):
        yield np.random.choice(items)
        # yield "/Users/justas/PycharmProjects/ugproject/img/img2.jpg"
        # yield tf.train.match_filenames_once("/Users/justas/PycharmProjects/ugproject/img/*.jpg")


class DataGenerator(object):
    def __init__(self, coord, max_queue_size=32, wait_time=1):
        # Change the shape of the input data here with the parameter shapes.
        self.wait_time = wait_time
        self.max_queue_size = max_queue_size
        self.queue = tf.FIFOQueue(max_queue_size, dtypes=[tf.string], shapes=[[]])
        self.queue_size = self.queue.size()
        self.threads = []
        self.coord = coord
        self.sample_placeholder = tf.constant("BLAH", dtype=tf.string)  # , shape=None)
        self.enqueue = self.queue.enqueue(self.sample_placeholder)

    def dequeue(self, num_elements):
        output = self.queue.dequeue_up_to(num_elements)
        return output

    def get_queue(self):
        return self.queue

    def producer_thread(self, sess):
        data_iter = load_data()
        while not self.coord.should_stop():
            img_file = next(data_iter)
            sess.run(self.enqueue, feed_dict={self.sample_placeholder: img_file})
            print("Added data")

            # img_file = tf.Variable(img_file, dtype=tf.string)
            # img_decoded = tf.image.decode_jpeg(img_file, channels=3)
            # img_decoded = tf.cast(img_decoded, tf.float32)
            # img_decoded = tf.reshape(img_decoded, shape=(224, 224, 3))

            # img_decoded = tf.cast(images, tf.float32)

            # img_decoded = tf.image.per_image_standardization(img_decoded)
            # img_decoded = tf.clip_by_value(img_decoded, -1.0, 1.0)
            # print('img-d', sess.run(img_decoded))
            # enq_op = self.queue.enqueue(img_file)
            # sess.run(self.queue.enqueue(img_file))

            # time.sleep(self.wait_time)
            time.sleep(np.random.uniform() * 2)
        print("Enqueue thread receives stop request.")

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.producer_thread, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
