import tensorflow as tf


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_loss(style_outputs, style_targets, style_weights):
    loss_per_layer = [tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()]
    style_loss = tf.reduce_sum(tf.multiply(tf.stack(loss_per_layer, 0), style_weights))
    return style_loss


def content_loss(content_outputs, content_targets, content_weights):
    loss_per_layer = [tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                      for name in content_outputs.keys()]
    content_loss = tf.reduce_sum(tf.multiply(loss_per_layer, content_weights))
    return content_loss


def style_content_loss(style_outputs, style_targets, style_weights, alpha, content_outputs, content_targets, content_weights, beta):
    return alpha * style_loss(style_outputs, style_targets, style_weights) + beta * content_loss(content_outputs, content_targets, content_weights)


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)
