"""
Module with definitions of prediction models
"""

import keras
import keras.applications


def get_pretrained_vgg_model(image_shape):
    """
    Builds a model based on pretrained VGG net
    :param image_shape: image shape
    :return:
    """

    input_layer = keras.layers.Input(shape=image_shape)

    x = keras.applications.VGG16(include_top=False, weights='imagenet')(input_layer)
    x = keras.layers.Convolution2D(1, 7, 7, activation='sigmoid', name='final_convolution')(x)
    x = keras.layers.Flatten()(x)

    model = keras.models.Model(input=input_layer, output=x)

    adam = keras.optimizers.Adam(lr=0.00001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model

