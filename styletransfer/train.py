import tensorflow as tf
import time
import matplotlib.pyplot as plt
import IPython.display as display

from util import clip_0_1, imshow


def train(loss_func, extractor, initial_gradients, epochs=20, steps_per_epoch=100):
    # variable to optimize
    transfer = tf.Variable(initial_gradients)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            loss = loss_func(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(transfer)
            print(".", end='')
        display.clear_output(wait=True)
        imshow(transfer.read_value())
        plt.title("Train step: {}".format(step))
        plt.show()

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    return transfer
