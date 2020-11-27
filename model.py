# %%
import argparse
import datetime
import os
import re
import PIL
import PIL.Image
import PIL.ImageChops

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import efficient_net

# %%
DATASET_PATH = "data"
BATCH_SIZE = 5
SHUFFLE_SIZE = 10000
AWE_W = 480
AWE_H = 360
AWE_C = 3
GROUP_NORM = 16

# %%


def load_dataset(basedir, images, segments):
    def transform(path):
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image)
        mask_path = tf.strings.regex_replace(path, images, segments)
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_png(mask)
        return image, mask

    images_dir = os.path.join(basedir, images)
    dataset = tf.data.Dataset.list_files(os.path.join(images_dir, '*.png'))
    return dataset.map(transform)


# %%
train = load_dataset(DATASET_PATH, 'train', 'trainannot')
test = load_dataset(DATASET_PATH, 'test', 'testannot')

# %%
train = train.cache().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
test = test.cache().batch(BATCH_SIZE)

# %%
efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
    include_top=False, dynamic_shape=True)
efficientnet_b0.trainable = False

# %%
x = inputs = tf.keras.layers.Input(shape=[AWE_H, AWE_W, AWE_C])

features, c5, c4, c3, c2, c1 = efficientnet_b0(x)


def conv_block(x, channels, kernel_size, stride=1):
    x = tf.keras.layers.Conv2D(
        channels, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = tfa.layers.GroupNormalization(GROUP_NORM)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def deconv_block(x, channels, kernel_size, stride=2):
    x = tf.keras.layers.Conv2DTranspose(
        channels, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = tfa.layers.GroupNormalization(GROUP_NORM)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def decoder(x, channels_in, channels_out):
    x = conv_block(x, channels_in // 4, 1)
    x = deconv_block(x, channels_in // 4, 3)
    x = conv_block(x, channels_in // 4, 3)
    x = conv_block(x, channels_out, 1)
    return x


def skip(x, c, channels):
    x = tf.keras.layers.Concatenate()([x, c])
    x = conv_block(x, channels, 1)
    return x


# x = c5
# x = decoder(x, 1280, 512)
# x = skip(x, c4, 512)
# x = decoder(x, 512, 256)
# x = skip(x, c3, 256)
x = c3
x = decoder(x, 256, 128)
x = skip(x, c2, 128)
x = decoder(x, 128, 64)
x = skip(x, c1, 64)
x = decoder(x, 64, 64)

# Head
x = deconv_block(x, 32, 3, 2)
x = conv_block(x, 32, 3, 2)

#deconv_block(x, 1, 2, 1)
x = tf.keras.layers.Conv2DTranspose(
    1, kernel_size=2, strides=1, padding="same", use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.nn.sigmoid)(x)

# %%
