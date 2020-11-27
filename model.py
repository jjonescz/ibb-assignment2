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

import efficient_net

# %%
DATASET_PATH = "data"

# %%
train_dir = os.path.join(DATASET_PATH, 'train')
train_images = list(map(
    lambda x: os.path.join(train_dir, x),
    os.listdir(train_dir)))
train_segments_dir = os.path.join(DATASET_PATH, 'trainannot')
train_segments = list(map(
    lambda x: os.path.join(train_segments_dir, x),
    os.listdir(train_segments_dir)))

# %%
image = PIL.Image.open(train_images[0])
bytes = np.asarray(PIL.Image.open(train_segments[0]))
mask = PIL.Image.fromarray(bytes * 255)

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

# %%
efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

# %%
