from concurrent.futures import ThreadPoolExecutor
import glob
import time
import rpyc

c = rpyc.connect("localhost", 18812)

tpe = ThreadPoolExecutor(max_workers=2)


def send_img(img_url):
    send_time = time.time()
    res = c.root.echo(img_url)
    print("delay (ms) ", 1000 * (time.time() - send_time))
    print("res", img_url, res)


# image_list = []
for filename in glob.glob("/Users/justas/PycharmProjects/ugproject/img/*.jpg"):  # assuming gif
    tpe.submit(send_img, filename)
    # send_img("/Users/justas/PycharmProjects/ugproject/img/000001.jpg")
