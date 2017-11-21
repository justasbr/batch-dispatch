import tensorflow as tf

s = tf.placeholder(tf.string)

simple_s = "/Users/justas/PycharmProjects/ugproject/img_big/a.jpg"

reader = tf.WholeFileReader()
q = tf.FIFOQueue(dtypes=[tf.string], capacity=5)
enq_op = q.enqueue(simple_s)

# filename = tf.constant("/Users/justas/PycharmProjects/ugproject/img/000001.jpg", dtype=tf.string, name="fname")
# filename = file_name  # tf.constant("/Users/justas/PycharmProjects/ugproject/img/000001.jpg", dtype=tf.string, name="fname")
# q = tf.train.string_input_producer([simple_s], shuffle=False)
#
_, img_val = reader.read(q)
img = tf.image.decode_jpeg(img_val, channels=3)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    init_op_local = tf.local_variables_initializer()

    sess.run(init_op)
    sess.run(init_op_local)

    # print(sess.run(s, feed_dict={s: "Hello world"}))
    #
    # print(sess.run(img, feed_dict={s: "/Users/justas/PycharmProjects/ugproject/img_big/a.jpg"}))

    coord = tf.train.Coordinator()
    threads_runners = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(sess.run(q.size()))
    print(sess.run(enq_op))
    print(sess.run(q.size()))

    print("A")
    # print(sess.run(img))
    print("B")

    coord.request_stop()
    coord.join(threads_runners)
