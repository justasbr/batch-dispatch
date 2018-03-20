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

def t_relu(layer):
    return F.relu(layer, inplace=True)

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=3, out_channels=64, kernel_size=(11, 11), stride=(4, 4), groups=1, bias=True)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=64, out_channels=192, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.conv2d_3 = self.__conv(2, name='conv2d_3', in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2d_4 = self.__conv(2, name='conv2d_4', in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2d_5 = self.__conv(2, name='conv2d_5', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.dense_1 = self.__dense(name = 'dense_1', in_features = 9216, out_features = 4096, bias = True)
        self.dense_2 = self.__dense(name = 'dense_2', in_features = 4096, out_features = 4096, bias = True)
        self.dense_3 = self.__dense(name = 'dense_3', in_features = 4096, out_features = 1000, bias = True)

    def forward(self, x):
        conv2d_1_pad    = F.pad(x, (3, 4, 3, 4))
        conv2d_1        = self.conv2d_1(conv2d_1_pad)
        activation_1    = t_relu(conv2d_1)
        max_pooling2d_1 = F.max_pool2d(activation_1, kernel_size=(3, 3), stride=(2, 2))
        conv2d_2_pad    = F.pad(max_pooling2d_1, (2, 2, 2, 2))
        conv2d_2        = self.conv2d_2(conv2d_2_pad)
        activation_2    = t_relu(conv2d_2)
        max_pooling2d_2 = F.max_pool2d(activation_2, kernel_size=(3, 3), stride=(2, 2))
        conv2d_3_pad    = F.pad(max_pooling2d_2, (1, 1, 1, 1))
        conv2d_3        = self.conv2d_3(conv2d_3_pad)
        activation_3    = t_relu(conv2d_3)
        conv2d_4_pad    = F.pad(activation_3, (1, 1, 1, 1))
        conv2d_4        = self.conv2d_4(conv2d_4_pad)
        activation_4    = t_relu(conv2d_4)
        conv2d_5_pad    = F.pad(activation_4, (1, 1, 1, 1))
        conv2d_5        = self.conv2d_5(conv2d_5_pad)
        activation_5    = t_relu(conv2d_5)
        max_pooling2d_3 = F.max_pool2d(activation_5, kernel_size=(3, 3), stride=(2, 2))
        flatten_1       = max_pooling2d_3.view(max_pooling2d_3.size(0), -1)
        dense_1         = self.dense_1(flatten_1)
        activation_6    = t_relu(dense_1)
        dense_2         = self.dense_2(activation_6)
        activation_7    = t_relu(dense_2)
        dense_3         = self.dense_3(activation_7)
        return dense_3


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

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
