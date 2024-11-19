import numpy as np
import tensorflow as tf

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20

train_path = "train/train"
test_path = "test/test"
def decode_image(image_path, label):
    image_path = str(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, label
def get_data_partitions_tf(ds, train_split, shuffle=True, shuffle_size=1000):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    n_train = int(train_split * ds_size)

    train_data = ds.take(n_train)
    val_data = ds.skip(n_train)

    return train_data, val_data