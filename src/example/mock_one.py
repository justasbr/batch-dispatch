import rpyc
from rpyc.utils.server import ThreadedServer
import os


def classify(img_path):
    file_name = os.path.basename(img_path)
    num, ext = file_name.split(".", 1)
    num = int(num) % 1000

    return num


class MyService(rpyc.Service):
    def exposed_echo(self, img_path):
        return classify(img_path=img_path)


if __name__ == "__main__":
    server = ThreadedServer(MyService, port=18812)
    print("Starting MOCK server!")
    server.start()
