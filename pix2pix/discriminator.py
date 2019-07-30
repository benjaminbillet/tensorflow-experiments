import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from generator import downsample_layer

# The shape of the output after the last layer is (batch_size, 30, 30, 1)
# Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).


def PatchDiscriminator():
    initializer = keras.initializers.RandomNormal(0.0, 0.02)

    inputs = keras.layers.Input(shape=[None, None, 3], name='input_image')
    target = keras.layers.Input(shape=[None, None, 3], name='target_image')

    model = keras.layers.concatenate([inputs, target])  # (bs, 256, 256, channels*2)

    down1 = downsample_layer(64, 4, apply_batchnorm=False)(model)  # (bs, 128, 128, 64)
    down2 = downsample_layer(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample_layer(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = keras.layers.Conv2D(512, 4, strides=1,
                               kernel_initializer=initializer,
                               use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm = keras.layers.BatchNormalization()(conv)
    leaky_relu = keras.layers.LeakyReLU()(batchnorm)
    zero_pad2 = keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = keras.layers.Conv2D(1, 4, strides=1,
                               kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return keras.Model(inputs=[inputs, target], outputs=last)


def discriminator_loss(discriminator_real, discriminator_generated):
    crossentropy_loss = keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = crossentropy_loss(tf.ones_like(discriminator_real), discriminator_real)
    generated_loss = crossentropy_loss(tf.zeros_like(discriminator_generated), discriminator_generated)
    return real_loss + generated_loss
