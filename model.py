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
BATCH_SIZE = 32
SHUFFLE_SIZE = 500
AWE_W = 480
AWE_H = 360
AWE_C = 3
GROUP_NORM = 16
EPOCHS = 1

# %%


def load_dataset(basedir, images, segments):
    def transform(path):
        def load(path):
            image = tf.io.read_file(path)
            image = tf.io.decode_png(image)
            return tf.cast(image, dtype=tf.float32)

        image = load(path)
        mask_path = tf.strings.regex_replace(path, images, segments)
        mask = load(mask_path)
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


def bce_dice_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return numerator / denominator

    return tf.losses.binary_crossentropy(y_true, y_pred) + 1 - dice_loss(y_true, y_pred)

# %%


class AWEMaskIoU(tf.metrics.Mean):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_mask = tf.reshape(tf.math.round(
            y_true) == 1, [-1, AWE_H * AWE_W])
        y_pred_mask = tf.reshape(tf.math.round(
            y_pred) == 1, [-1, AWE_H * AWE_W])

        intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
        union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

        intersection = tf.reduce_sum(
            tf.cast(intersection_mask, tf.float32), axis=1)
        union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

        iou = tf.where(union == 0, 1., intersection / union)
        return super().update_state(iou, sample_weight)


# %%
model = tf.keras.Model(inputs=inputs, outputs=x)

model.compile(
    optimizer=tf.optimizers.RMSprop(),
    loss=bce_dice_loss,
    metrics=[AWEMaskIoU(name="accuracy")],
)

# %%
model.fit(
    train,
    epochs=EPOCHS,
    validation_data=test
)

# %%
