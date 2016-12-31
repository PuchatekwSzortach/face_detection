"""
Module with definitions of prediction models
"""

import keras
import keras.applications


def get_pretrained_vgg_model(image_shape):
    """
    Builds a model based on pretrained VGG net
    :param image_shape: image shape
    :return: keras model
    """

    expected_image_shape = (224, 224, 3)

    if image_shape != expected_image_shape:

        message = "Input image is specified to be {}, but this model is designed to work with inputs of shape {}"\
            .format(image_shape, expected_image_shape)

        raise ValueError(message)

    input_layer = keras.layers.Input(shape=image_shape)

    x = keras.applications.VGG16(include_top=False, weights='imagenet')(input_layer)
    x = keras.layers.Convolution2D(1, 7, 7, activation='sigmoid', name='final_convolution')(x)
    x = keras.layers.Flatten()(x)

    model = keras.models.Model(input=input_layer, output=x)

    adam = keras.optimizers.Adam(lr=0.00001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_medium_scale_model(image_shape):
    """
    Builds a model intended to work on crops of size 100x100. Significantly smaller complexity than VGG net,
    but similar design
    :param image_shape: image shape
    :return: keras model
    """

    expected_image_shape = (100, 100, 3)

    if image_shape != expected_image_shape:
        message = "Input image is specified to be {}, but this model is designed to work with inputs of shape {}" \
            .format(image_shape, expected_image_shape)

        raise ValueError(message)

    input_layer = keras.layers.Input(shape=image_shape)

    # Block 1
    x = keras.layers.Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='block1_conv1')(input_layer)
    x = keras.layers.Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='block2_conv1')(x)
    x = keras.layers.Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 4
    x = keras.layers.Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block4_conv1')(x)
    x = keras.layers.Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block4_conv2')(x)
    x = keras.layers.Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = keras.layers.Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv1')(x)
    x = keras.layers.Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv2')(x)
    x = keras.layers.Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = keras.layers.Convolution2D(1, 6, 6, activation='sigmoid', name='final_convolution')(x)
    x = keras.layers.Flatten()(x)

    model = keras.models.Model(input=input_layer, output=x)

    adam = keras.optimizers.Adam(lr=0.000001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model




