import rpyc
from rpyc.utils.server import ThreadedServer

# from main_tf import print_activations
from numpy import prod
from skimage import io, transform

from alexnet_torch import alexnet
import torch
import torchvision
from torch.autograd import Variable

net = alexnet(pretrained=True)
transform = torchvision.transforms.ToTensor()


def classify(img_path):
    img_data = transform(io.imread(img_path))
    img_data = Variable(img_data)

    batch = torch.stack([img_data])
    output = net.forward(batch)
    max_val, max_index = torch.max(output, 1)
    max_index = max_index.data[0]
    return max_index


class MyService(rpyc.Service):
    def __init__(self, conn):
        super().__init__(conn)
        print("MASTER INIT")

    def exposed_echo(self, img_path):
        return classify(img_path)


if __name__ == "__main__":
    server = ThreadedServer(MyService, port=18812)
    print("Starting Torch server!")
    server.start()
