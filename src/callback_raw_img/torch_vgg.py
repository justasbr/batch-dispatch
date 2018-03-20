import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

def t_relu(layer, inplace=True):
    return F.relu(layer,inplace=True)


class KitModel(nn.Module):


    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.block1_conv1 = self.__conv(2, name='block1_conv1', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block1_conv2 = self.__conv(2, name='block1_conv2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block2_conv1 = self.__conv(2, name='block2_conv1', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block2_conv2 = self.__conv(2, name='block2_conv2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_conv1 = self.__conv(2, name='block3_conv1', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_conv2 = self.__conv(2, name='block3_conv2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block3_conv3 = self.__conv(2, name='block3_conv3', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_conv1 = self.__conv(2, name='block4_conv1', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_conv2 = self.__conv(2, name='block4_conv2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block4_conv3 = self.__conv(2, name='block4_conv3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_conv1 = self.__conv(2, name='block5_conv1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_conv2 = self.__conv(2, name='block5_conv2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.block5_conv3 = self.__conv(2, name='block5_conv3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.fc1 = self.__dense(name = 'fc1', in_features = 25088, out_features = 4096, bias = True)
        self.fc2 = self.__dense(name = 'fc2', in_features = 4096, out_features = 4096, bias = True)
        self.predictions = self.__dense(name = 'predictions', in_features = 4096, out_features = 1000, bias = True)

    def forward(self, x):
        block1_conv1_pad = F.pad(x, (1, 1, 1, 1))
        block1_conv1    = self.block1_conv1(block1_conv1_pad)
        block1_conv1_activation = t_relu(block1_conv1, inplace=True)
        block1_conv2_pad = F.pad(block1_conv1_activation, (1, 1, 1, 1))
        block1_conv2    = self.block1_conv2(block1_conv2_pad)
        block1_conv2_activation = t_relu(block1_conv2, inplace=True)
        block1_pool     = F.max_pool2d(block1_conv2_activation, kernel_size=(2, 2), stride=(2, 2))
        block2_conv1_pad = F.pad(block1_pool, (1, 1, 1, 1))
        block2_conv1    = self.block2_conv1(block2_conv1_pad)
        block2_conv1_activation = t_relu(block2_conv1, inplace=True)
        block2_conv2_pad = F.pad(block2_conv1_activation, (1, 1, 1, 1))
        block2_conv2    = self.block2_conv2(block2_conv2_pad)
        block2_conv2_activation = t_relu(block2_conv2, inplace=True)
        block2_pool     = F.max_pool2d(block2_conv2_activation, kernel_size=(2, 2), stride=(2, 2))
        block3_conv1_pad = F.pad(block2_pool, (1, 1, 1, 1))
        block3_conv1    = self.block3_conv1(block3_conv1_pad)
        block3_conv1_activation = t_relu(block3_conv1, inplace=True)
        block3_conv2_pad = F.pad(block3_conv1_activation, (1, 1, 1, 1))
        block3_conv2    = self.block3_conv2(block3_conv2_pad)
        block3_conv2_activation = t_relu(block3_conv2, inplace=True)
        block3_conv3_pad = F.pad(block3_conv2_activation, (1, 1, 1, 1))
        block3_conv3    = self.block3_conv3(block3_conv3_pad)
        block3_conv3_activation = t_relu(block3_conv3, inplace=True)
        block3_pool     = F.max_pool2d(block3_conv3_activation, kernel_size=(2, 2), stride=(2, 2))
        block4_conv1_pad = F.pad(block3_pool, (1, 1, 1, 1))
        block4_conv1    = self.block4_conv1(block4_conv1_pad)
        block4_conv1_activation = t_relu(block4_conv1, inplace=True)
        block4_conv2_pad = F.pad(block4_conv1_activation, (1, 1, 1, 1))
        block4_conv2    = self.block4_conv2(block4_conv2_pad)
        block4_conv2_activation = t_relu(block4_conv2, inplace=True)
        block4_conv3_pad = F.pad(block4_conv2_activation, (1, 1, 1, 1))
        block4_conv3    = self.block4_conv3(block4_conv3_pad)
        block4_conv3_activation = t_relu(block4_conv3, inplace=True)
        block4_pool     = F.max_pool2d(block4_conv3_activation, kernel_size=(2, 2), stride=(2, 2))
        block5_conv1_pad = F.pad(block4_pool, (1, 1, 1, 1))
        block5_conv1    = self.block5_conv1(block5_conv1_pad)
        block5_conv1_activation = t_relu(block5_conv1, inplace=True)
        block5_conv2_pad = F.pad(block5_conv1_activation, (1, 1, 1, 1))
        block5_conv2    = self.block5_conv2(block5_conv2_pad)
        block5_conv2_activation = t_relu(block5_conv2, inplace=True)
        block5_conv3_pad = F.pad(block5_conv2_activation, (1, 1, 1, 1))
        block5_conv3    = self.block5_conv3(block5_conv3_pad)
        block5_conv3_activation = t_relu(block5_conv3, inplace=True)
        block5_pool     = F.max_pool2d(block5_conv3_activation, kernel_size=(2, 2), stride=(2, 2))
        flatten         = block5_pool.view(block5_pool.size(0), -1)
        fc1             = self.fc1(flatten)
        fc1_activation  = t_relu(fc1)
        fc2             = self.fc2(fc1_activation)
        fc2_activation  = t_relu(fc2)
        predictions     = self.predictions(fc2_activation)
        predictions_activation = F.softmax(predictions,dim=1)
        return predictions_activation


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
