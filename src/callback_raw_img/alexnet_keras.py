from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import h5py


# AlexNet with batch normalization in Keras
# input image is 224x224

def create_model_alex():
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=11, strides=(4, 4),
                            padding="same", input_shape=(224, 224, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Convolution2D(filters=192, kernel_size=5, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Convolution2D(filters=384, kernel_size=3, padding="same", activation="relu"))

    model.add(Convolution2D(filters=256, kernel_size=3, padding="same", activation="relu"))

    model.add(Convolution2D(filters=256, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer="normal", activation="relu", use_bias=True))

    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer="normal", activation="relu", use_bias=True))

    model.add(Dense(1000, kernel_initializer="normal", use_bias=True))
    model.add(Activation('softmax'))

    print(model.summary())

    #         nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),

    #         nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),

    #         nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),

    #         nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),

    #         nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #
    #         nn.Dropout(),
    #         nn.Linear(256 * 6 * 6, 4096),
    #         nn.ReLU(inplace=True),

    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(inplace=True),

    #         nn.Linear(4096, num_classes),
    #
    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     return x

    # model.add(Flatten())  # , input_dim)
    # model.add(Dense(1000, kernel_initializer="normal"))
    return model


def create_model_vgg16():
    return VGG16(weights='imagenet', include_top=True)
