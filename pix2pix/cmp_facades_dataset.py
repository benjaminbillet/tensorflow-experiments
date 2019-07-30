import tensorflow as tf
import tensorflow.keras as keras
import os

from util import random_jitter, normalize, resize

IMG_WIDTH = 256
IMG_HEIGHT = 256


def download_dataset():
    _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    path_to_zip = keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
    return os.path.join(os.path.dirname(path_to_zip), 'facades/')


def load_dataset(buffer_size):
    path = download_dataset()

    # Input Pipeline
    train_dataset = tf.data.Dataset.list_files(path+'train/*.jpg')
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.map(load_train_image,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(1)

    test_dataset = tf.data.Dataset.list_files(path+'test/*.jpg')
    # shuffling so that for every epoch a different image is generated
    # to predict and display the progress of our model.
    train_dataset = train_dataset.shuffle(buffer_size)
    test_dataset = test_dataset.map(load_test_image)
    test_dataset = test_dataset.batch(1)

    return train_dataset, test_dataset


def load_dataset_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    width = tf.shape(image)[1]

    width = width // 2
    real_image = image[:, :width, :]
    input_image = image[:, width:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def load_train_image(image_file):
    input_image, real_image = load_dataset_image(image_file)
    input_image, real_image = random_jitter(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_test_image(image_file):
    input_image, real_image = load_dataset_image(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image
