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
from keras.models import load_model


def create_model(model):
    print("Loading Keras model from a h5 file.")
    model_folder = "keras_frozen/"
    if model == "alexnet":
        return create_model_alex()
        # return load_model(model_folder + "keras_alexnet.h5", compile=False)
    elif model == "vgg":
        return load_model(model_folder + "keras_vgg.h5", compile=False)
    elif model == "inception":
        return load_model(model_folder + "keras_inception.h5", compile=False)
    elif model == "resnet":
        return load_model(model_folder + "keras_resnet.h5", compile=False)
    else:
        raise RuntimeError("No such model exists: " + str(model))


def create_model_old(model):
    if model == "alexnet":
        return create_model_alex()
    elif model == "vgg":
        return create_model_vgg16()
    elif model == "inception":
        return create_model_inception_v3()
    elif model == "resnet":
        return create_model_resnet50()
    else:
        raise RuntimeError("No such model exists: " + str(model))


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
    model.add(Dense(4096, kernel_initializer="normal", use_bias=True))
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer="normal", use_bias=True))
    model.add(Activation('relu'))

    model.add(Dense(1000, kernel_initializer="normal", use_bias=True))
    model.add(Activation('softmax'))

    model.load_weights('keras_frozen/keras_alexnet_weights.h5')

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
