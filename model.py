import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

# Parameters
DATASET_PATH = "data"
BATCH_SIZE = 10
SHUFFLE_SIZE = 500
AWE_W = 480
AWE_H = 352  # divisible by 32
AWE_C = 3
GROUP_NORM = 16
EPOCHS = 35
EXP_ID = "final"  # subfolder inside `out/` with saved weights
TRAIN = False  # `True` = train, `False` = load saved checkpoints
OUT_DIR = os.path.join("out", EXP_ID)


def load_dataset(basedir, images, segments):
    def transform(path):
        def load(path, channels):
            # Read and resize PNG.
            image = tf.io.read_file(path)
            image = tf.io.decode_png(image, channels=channels)
            return tf.image.resize(image, (AWE_H, AWE_W))

        # Load image and its gold mask.
        image = load(path, AWE_C)
        mask_path = tf.strings.regex_replace(path, images, segments)
        mask = load(mask_path, 1)
        return image, mask

    # Load all images and their masks.
    images_dir = os.path.join(basedir, images)
    dataset = tf.data.Dataset.list_files(os.path.join(images_dir, '*.png'))
    return dataset.map(transform)


# Load AWE dataset.
train_orig = load_dataset(DATASET_PATH, 'train', 'trainannot')
test_orig = load_dataset(DATASET_PATH, 'test', 'testannot')

# Split into train, dev and test datasets.
train = train_orig.take(500).cache().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
dev = train_orig.skip(500).cache().batch(BATCH_SIZE)
test = test_orig.cache().batch(BATCH_SIZE)

# Load (or download) EfficientNet-B0.
efficientnet_b0 = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=(AWE_H, AWE_W, AWE_C))
efficientnet_b0.trainable = False

# Find outputs of EfficientNet we care about.
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

# Construct our CNN model.
x = inputs = tf.keras.layers.Input(shape=(AWE_H, AWE_W, AWE_C))

# Encoder
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


def skip(x, c, channels):
    x = tf.keras.layers.Concatenate()([x, c])
    x = conv_block(x, channels, 1)
    return x


# Decoder
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
x = tf.keras.layers.Conv2DTranspose(
    1, kernel_size=2, strides=1, padding="same", use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.nn.sigmoid)(x)


def bce_dice_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return numerator / denominator

    return tf.losses.binary_crossentropy(y_true, y_pred) + 1 - dice_loss(y_true, y_pred)


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


# Create model.
model = tf.keras.Model(inputs=inputs, outputs=x)

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=bce_dice_loss,
    metrics=[AWEMaskIoU(name="accuracy")]
)

# Create callback which will save checkpoints during training.
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(OUT_DIR, 'train-{epoch:04d}.ckpt'),
    save_weights_only=True,
    verbose=1
)

# Train the model and save metric values (or load saved metric values).
history_path = os.path.join(OUT_DIR, 'history.json')
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

# Save final weights (or load them).
weights_path = os.path.join(OUT_DIR, 'weights.h5')
if TRAIN:
    model.save_weights(weights_path)
else:
    model.load_weights(weights_path)

# Plot loss evolution during training.
loss_history = train_history_dict['loss']
val_loss_history = train_history_dict['val_loss']
plt.plot(loss_history, label='training')
plt.plot(val_loss_history, label='development')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('BCE dice loss')
plt.title('Loss during training')
plt.savefig('figures/loss.pdf', bbox_inches='tight', pad_inches=0)
plt.clf()

# Plot IoU evolution during training.
accuracy_history = train_history_dict['accuracy']
val_accuracy_history = train_history_dict['val_accuracy']
plt.plot(accuracy_history, label='training')
plt.plot(val_accuracy_history, label='development')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('IoU')
plt.title('IoU during training')
plt.savefig('figures/iou.pdf', bbox_inches='tight', pad_inches=0)
plt.clf()

# Run the model on development data.
predictions = model.predict(dev, verbose=1)


def compute_iou(gold, pred):
    iou = AWEMaskIoU()
    iou.update_state(gold, pred)
    return iou.result()


# Compute IoU on development data.
prediction_ious = np.array([
    compute_iou(gold, pred)
    for (_, gold), pred in zip(dev.unbatch(), predictions)
])

# Find three best and three worst predictions.
K = 3
worst = prediction_ious.argsort()[:K]
best = (-prediction_ious).argsort()[:K]


def plot_examples(name, indices):
    rows = len(indices)
    plt.figure(figsize=(24, 6 * rows))
    for row, i in enumerate(indices):
        # Find data at the specified index.
        image, gold_mask = None, None
        for image, gold_mask in dev.unbatch().skip(i).take(1):
            break
        pred_mask = predictions[i]

        # Plot original image.
        ax_im = plt.subplot(rows, 3, 3 * row + 1)
        ax_im.imshow(image.numpy().astype('uint8'))
        ax_im.axis('off')
        if row == 0:
            ax_im.set_title('Original', fontsize=40)

        # Plot gold mask.
        ax_g = plt.subplot(rows, 3, 3 * row + 2)
        ax_g.imshow(gold_mask.numpy(), cmap='gray', vmin=0, vmax=1)
        ax_g.axis('off')
        if row == 0:
            ax_g.set_title('Gold', fontsize=40)

        # Plot predicted mask.
        ax_p = plt.subplot(rows, 3, 3 * row + 3)
        ax_p.imshow(pred_mask.round(), cmap='gray', vmin=0, vmax=1)
        ax_p.axis('off')
        if row == 0:
            ax_p.set_title('Predicted', fontsize=40)
    plt.savefig(f'figures/{name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.clf()


# Plot three best and three worst predictions.
plot_examples('examples-best', best)
plot_examples('examples-worst', worst)

# Plot IoU histogram.
plt.hist(prediction_ious)
plt.title('IoU in development dataset')
plt.xlabel('intersection over union (IoU)')
plt.ylabel('number of images')
plt.savefig('figures/iou-hist.pdf', bbox_inches='tight', pad_inches=0)
plt.clf()

# Evaluate test data.
loss, iou = model.evaluate(test)
print(f'IoU on test data: {iou:.2%}')
