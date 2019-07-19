from model import StyleTransferModel
from losses import style_content_loss, total_variation_loss
from train import train


def transfer_style(content_img, style_img, initial_gradients=None, alpha=1e1, beta=1e-2, total_variation_weight=1e8, style_weights=[1.0, 1.0, 1.0, 1.0, 1.0], epochs=20):
    content_layers = ['block5_conv2']
    content_weights = [1.0]

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    return transfer_style_advanced(content_img, style_img, initial_gradients, content_layers, style_layers, content_weights, style_weights, alpha, beta, total_variation_weight, epochs)


def merge_styles(coarse_style, fine_style, initial_gradients=None, style_weights=[1.0, 1.0], beta=1e0, total_variation_weight=1e8, epochs=20):
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1']

    return transfer_style_advanced(coarse_style, fine_style, initial_gradients, content_layers, style_layers, [1.0], style_weights, 0, beta, total_variation_weight, epochs)


def transfer_style_advanced(content_img, style_img, initial_gradients=None, content_layers=['block5_conv2'], style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'], content_weights=[1.0], style_weights=[1.0, 1.0, 1.0, 1.0, 1.0], alpha=1e1, beta=1e-2, total_variation_weight=1e8, epochs=20):
    extractor = StyleTransferModel(style_layers, content_layers)

    content_targets = extractor(content_img)['content']
    style_targets = extractor(style_img)['style']

    # normalize weights (is it good?)
    style_weights = [w/sum(style_weights) for w in style_weights]
    content_weights = [w/sum(content_weights) for w in content_weights]

    if initial_gradients is None:
        initial_gradients = content_img

    def loss_func(image):
        outputs = extractor(image)
        loss = style_content_loss(outputs['content'], content_targets, content_weights,
                                  alpha, outputs['style'], style_targets, style_weights, beta)
        loss += total_variation_weight*total_variation_loss(image)
        return loss

    transferred = train(loss_func, extractor, initial_gradients, epochs=epochs)
    return transferred
