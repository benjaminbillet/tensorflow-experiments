import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def load_img(path_to_img, max_dim=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def download_img(filename, url_to_img):
    img_path = tf.keras.utils.get_file(filename, url_to_img)
    return load_img(img_path)


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


def apply_lum(content_img, style_img):
    style_yuv = tf.image.rgb_to_yuv(style_img)
    style_channels = tf.unstack(style_yuv, axis=-1)

    content_yuv = tf.image.rgb_to_yuv(content_img)
    content_channels = tf.unstack(content_yuv, axis=-1)

    new_img = tf.stack([style_channels[0], content_channels[1], content_channels[2]], axis=-1)
    return tf.image.yuv_to_rgb(new_img)


def match_lum(content_img, style_img):
    content_yuv = tf.image.rgb_to_yuv(content_img)
    content_channels = tf.unstack(content_yuv, axis=-1)
    content_y_mean = tf.math.reduce_mean(content_channels[0])
    content_y_stddev = tf.math.reduce_std(content_channels[0])

    style_yuv = tf.image.rgb_to_yuv(style_img)
    style_channels = tf.unstack(style_yuv, axis=-1)
    style_y_mean = tf.math.reduce_mean(style_channels[0])
    style_y_stddev = tf.math.reduce_std(style_channels[0])

    stddev_ratio = content_y_stddev / style_y_stddev
    style_y_corrected = tf.map_fn(
        lambda y: stddev_ratio * (y - style_y_mean) + content_y_mean,
        style_channels[0])

    new_img = tf.stack([style_y_corrected, style_channels[1], style_channels[2]], axis=-1)
    return tf.image.yuv_to_rgb(new_img)
