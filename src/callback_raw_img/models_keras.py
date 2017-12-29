from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import h5py


# AlexNet with batch normalization in Keras
# input image is 224x224

def create_model_alex():
    model = Sequential()

    # FEATURES
    model.add(Convolution2D(filters=64, kernel_size=11, strides=(4, 4),
                            padding="same", input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Convolution2D(filters=192, kernel_size=5, padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Convolution2D(filters=384, kernel_size=3, padding="same"))
    model.add(Activation('relu'))

    model.add(Convolution2D(filters=256, kernel_size=3, padding="same"))
    model.add(Activation('relu'))

    model.add(Convolution2D(filters=256, kernel_size=3, padding="same"))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # CLASSIFIER
    model.add(Flatten())

    # model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer="normal",  use_bias=True))
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer="normal", use_bias=True))
    model.add(Activation('relu'))

    model.add(Dense(1000, kernel_initializer="normal", use_bias=True))
    # model.add(Activation('softmax'))

    print(model.summary())

    return model


def create_model_vgg16():
    model = VGG16(weights='imagenet', include_top=True)
    print(model.summary())
    return model


def create_model_resnet50():
    model = ResNet50(weights='imagenet', include_top=True)
    print(model.summary())
    return model


def create_model_inception_v3():
    model = InceptionV3(weights='imagenet', include_top=True)
    print(model.summary())
    return model
