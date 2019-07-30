import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image, height, width):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, height, width, 3])
    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image, height, width):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, height + 30, width + 30)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image, height, width)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def save_img(image, name, folder='output'):
    os.makedirs(folder, mode=0o777, exist_ok=True)
    file_name = os.path.join(folder, name)
    mpl.image.imsave(file_name, image)
