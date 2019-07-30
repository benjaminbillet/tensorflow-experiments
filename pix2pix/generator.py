import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def downsample_layer(filters, size, strides=2, apply_batchnorm=True):
    layer = keras.Sequential()

    initializer = keras.initializers.RandomNormal(0.0, 0.02)
    conv = keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                               kernel_initializer=initializer, use_bias=False)

    layer.add(conv)

    if apply_batchnorm:
        layer.add(keras.layers.BatchNormalization())

    layer.add(keras.layers.LeakyReLU())
    return layer


def upsample_layer(filters, size, strides=2, apply_dropout=False):
    layer = keras.Sequential()

    initializer = keras.initializers.RandomNormal(0.0, 0.02)
    conv = keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',
                                        kernel_initializer=initializer, use_bias=False)

    layer.add(conv)

    layer.add(keras.layers.BatchNormalization())

    if apply_dropout:
        layer.add(keras.layers.Dropout(0.5))

    layer.add(keras.layers.ReLU())
    return layer


def UnetGenerator(output_channels=3):
    encoder_layers = [
        downsample_layer(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample_layer(128, 4),  # (bs, 64, 64, 128)
        downsample_layer(256, 4),  # (bs, 32, 32, 256)
        downsample_layer(512, 4),  # (bs, 16, 16, 512)
        downsample_layer(512, 4),  # (bs, 8, 8, 512)
        downsample_layer(512, 4),  # (bs, 4, 4, 512)
        downsample_layer(512, 4),  # (bs, 2, 2, 512)
        downsample_layer(512, 4),  # (bs, 1, 1, 512)
    ]

    decoder_layers = [
        upsample_layer(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample_layer(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample_layer(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample_layer(512, 4),  # (bs, 16, 16, 1024)
        upsample_layer(256, 4),  # (bs, 32, 32, 512)
        upsample_layer(128, 4),  # (bs, 64, 64, 256)
        upsample_layer(64, 4),  # (bs, 128, 128, 128)
    ]

    concat = keras.layers.Concatenate()
    inputs = keras.layers.Input(shape=[None, None, 3])

    model = inputs

    # Downsampling through the model
    skips = []
    for layer in encoder_layers:
        model = layer(model)
        skips.append(model)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for layer, skip in zip(decoder_layers, skips):
        model = layer(model)
        model = concat([model, skip])

    initializer = keras.initializers.RandomNormal(0.0, 0.02)
    last = keras.layers.Conv2DTranspose(output_channels, 4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        activation='tanh')  # (bs, 256, 256, 3)

    model = last(model)

    return keras.Model(inputs=inputs, outputs=model)


def generator_loss(discriminator_generated, generated, target, lambda_term=100):
    crossentropy_loss = keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = crossentropy_loss(tf.ones_like(discriminator_generated), discriminator_generated)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - generated))
    return gan_loss + (lambda_term * l1_loss)


def generate_images(model, test_input, target):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Actual Image', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
