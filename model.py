# %%
import argparse
import datetime
import os
import re
import PIL
import PIL.Image
import PIL.ImageChops
import matplotlib.pyplot as plt
import json

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# %%
DATASET_PATH = "data"
BATCH_SIZE = 10
SHUFFLE_SIZE = 500
AWE_W = 480
AWE_H = 352  # divisible by 32
AWE_C = 3
GROUP_NORM = 16
EPOCHS = 35
EXP_ID = "more-epochs"
TRAIN = False  # `True` = train, `False` = load saved checkpoints

# %%


def load_dataset(basedir, images, segments):
    def transform(path):
        def load(path, channels):
            image = tf.io.read_file(path)
            image = tf.io.decode_png(image, channels=channels)
            return tf.image.resize(image, (AWE_H, AWE_W))

        image = load(path, AWE_C)
        mask_path = tf.strings.regex_replace(path, images, segments)
        mask = load(mask_path, 1)
        return image, mask

    images_dir = os.path.join(basedir, images)
    dataset = tf.data.Dataset.list_files(os.path.join(images_dir, '*.png'))
    return dataset.map(transform)


# %%
train_orig = load_dataset(DATASET_PATH, 'train', 'trainannot')
test_orig = load_dataset(DATASET_PATH, 'test', 'testannot')

# %%
train = train_orig.take(500).cache().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
dev = train_orig.skip(500).cache().batch(BATCH_SIZE)
test = test_orig.cache().batch(BATCH_SIZE)

# %%
efficientnet_b0 = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=(AWE_H, AWE_W, AWE_C))
efficientnet_b0.trainable = False

# %%
efficientnet_layer_names = [
    'top_activation',
    'block5c_add',
    'block3b_add',
    'block2b_add',
    'block1a_project_bn'
]
efficientnet_outputs = [efficientnet_b0.get_layer(
    name=name).output for name in efficientnet_layer_names]

efficientnet_model = tf.keras.Model(
    inputs=efficientnet_b0.input, outputs=efficientnet_outputs)
efficientnet_model.trainable = False

# %%
x = inputs = tf.keras.layers.Input(shape=(AWE_H, AWE_W, AWE_C))

[c5, c4, c3, c2, c1] = efficientnet_model(x)


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


x = c5
x = deconv_block(x, 512, 3)
x = skip(x, c4, 256)
x = deconv_block(x, 256, 3)
x = skip(x, c3, 128)
x = deconv_block(x, 128, 3)
x = skip(x, c2, 64)
x = deconv_block(x, 64, 3)
x = skip(x, c1, 32)
x = deconv_block(x, 32, 3)

# Head
# x = deconv_block(x, 32, 3, 2)
# x = conv_block(x, 32, 3, 2)

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
LOG_DIR = os.path.join("logs", EXP_ID)
# tb_callback = tf.keras.callbacks.TensorBoard(
#     LOG_DIR, histogram_freq=1, update_freq=100, profile_batch=0)

# %%
model = tf.keras.Model(inputs=inputs, outputs=x)

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=bce_dice_loss,
    metrics=[AWEMaskIoU(name="accuracy")]
)

# %%
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(LOG_DIR, 'train-{epoch:04d}.ckpt'),
    save_weights_only=True,
    verbose=1
)

# %%
history_path = os.path.join(LOG_DIR, 'history.json')
if TRAIN:
    train_history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=dev,
        callbacks=[cp_callback]
    )
    train_history_dict = train_history.history
    json.dump(train_history_dict, open(history_path, 'w'))
else:
    train_history_dict = json.load(open(history_path, 'r'))

# %%
weights_path = os.path.join(LOG_DIR, 'weights.h5')
if TRAIN:
    model.save_weights(weights_path)
else:
    model.load_weights(weights_path)

# %%
loss_history = train_history_dict['loss']
val_loss_history = train_history_dict['val_loss']
plt.plot(loss_history, label='training')
plt.plot(val_loss_history, label='development')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('BCE dice loss')
plt.title('Loss during training')
plt.savefig('figures/loss.pdf', bbox_inches='tight', pad_inches=0)

# %%
accuracy_history = train_history_dict['accuracy']
val_accuracy_history = train_history_dict['val_accuracy']
plt.plot(accuracy_history, label='training')
plt.plot(val_accuracy_history, label='development')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('IoU')
plt.title('IoU during training')
plt.savefig('figures/iou.pdf', bbox_inches='tight', pad_inches=0)

# %%
predictions = model.predict(dev)

# %%


def compute_iou(gold, pred):
    iou = AWEMaskIoU()
    iou.update_state(gold, pred)
    return iou.result()


prediction_ious = np.array([
    compute_iou(gold, pred)
    for (_, gold), pred in zip(dev.unbatch(), predictions)
])

# %%
K = 3
worst = prediction_ious.argsort()[:K]
best = (-prediction_ious).argsort()[:K]

# %%


def plot_examples(name, indices):
    rows = len(indices)
    plt.figure(figsize=(24, 6 * rows))
    for row, i in enumerate(indices):
        image, gold_mask = None, None
        for image, gold_mask in dev.unbatch().skip(i).take(1):
            break
        pred_mask = predictions[i]

        ax_im = plt.subplot(rows, 3, 3 * row + 1)
        ax_im.imshow(image.numpy().astype('uint8'))
        ax_im.axis('off')
        if row == 0:
            ax_im.set_title('Original', fontsize=40)

        ax_g = plt.subplot(rows, 3, 3 * row + 2)
        ax_g.imshow(gold_mask.numpy(), cmap='gray', vmin=0, vmax=1)
        ax_g.axis('off')
        if row == 0:
            ax_g.set_title('Gold', fontsize=40)

        ax_p = plt.subplot(rows, 3, 3 * row + 3)
        ax_p.imshow(pred_mask.round(), cmap='gray', vmin=0, vmax=1)
        ax_p.axis('off')
        if row == 0:
            ax_p.set_title('Predicted', fontsize=40)
    plt.savefig(f'figures/{name}.pdf', bbox_inches='tight', pad_inches=0)


# %%
plot_examples('examples-best', best)
plot_examples('examples-worst', worst)

# %%
plt.hist(prediction_ious)
plt.title('IoU in development dataset')
plt.xlabel('intersection over union (IoU)')
plt.ylabel('number of images')
plt.savefig('figures/iou-hist.pdf', bbox_inches='tight', pad_inches=0)

# %%
