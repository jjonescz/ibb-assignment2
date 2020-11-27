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
efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

# %%
