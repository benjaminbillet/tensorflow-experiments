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


def upscale(img, target_shape):
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    long_dim = max(shape)
    target_long_dim = max(target_shape)
    scale = target_long_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, target_shape)
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


def matrix_pow(m, p):
    s, u, v = tf.linalg.svd(m)
    s_pow = tf.linalg.diag(tf.pow(s, p))
    result = tf.matmul(tf.matmul(u, s_pow), v, transpose_b=True)
    return result


def covariance_matrix(img, means):
    flattened = tf.reshape(img, [-1, 3])
    flattened = tf.subtract(flattened, [means])
    cov = tf.matmul(flattened, flattened, transpose_a=True)
    cov = tf.divide(cov, flattened.shape[0])
    return cov


def match_color_histogram(source_img, dest_img):
    dest_channels = tf.unstack(dest_img, axis=-1)
    dest_means = [tf.math.reduce_mean(dest_channels[i]) for i in range(len(dest_channels))]

    source_channels = tf.unstack(source_img, axis=-1)
    source_means = [tf.math.reduce_mean(source_channels[i]) for i in range(len(source_channels))]

    dest_cov = covariance_matrix(dest_img[0], dest_means)
    source_cov = covariance_matrix(source_img[0], source_means)

    beta = tf.matmul(matrix_pow(source_cov, 0.5), matrix_pow(dest_cov, -0.5))
    alpha = tf.matmul(beta, tf.reshape(dest_means, [3, 1]))
    alpha = source_means - tf.reshape(alpha, [1, 3])

    # histogram matched version, for each pixel p, we have p' = alpha*p+beta
    flattened = tf.reshape(dest_img, [-1, 3])
    pixels = tf.unstack(flattened, axis=-1)
    pixels = tf.matmul(beta, pixels)
    pixels = tf.add(tf.stack(pixels, -1), alpha)

    # this is equivalent, but very slow
    # pixels = tf.map_fn(
    #     lambda p: tf.add(tf.reshape(tf.matmul(beta, tf.reshape(p, [3, 1])), [1, 3]), alpha),
    #     flattened)

    new_img = tf.reshape(pixels, dest_img.shape)
    return new_img
