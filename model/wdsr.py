# import tensorflow_addons as tfa

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

from tensorflow.keras.initializers import VarianceScaling

from model.common import normalize, denormalize, pixel_shuffle


def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    # Create a standard Conv2D layer
    conv_layer = Conv2D(filters, kernel_size, padding=padding, activation=None, **kwargs)
    
    # Initialize weights using VarianceScaling
    conv_layer.kernel_initializer = VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
    
    # Apply weight normalization to the Conv2D layer
    def custom_forward(x):
        # Apply convolution operation
        output = conv_layer(x)
        
        # Calculate the norm of the weights
        weight_norm = K.sqrt(K.sum(K.square(conv_layer.kernel)))
        
        # Normalize the weights
        normalized_weights = conv_layer.kernel / weight_norm
        
        # Update the Conv2D layer's kernel with the normalized weights
        conv_layer.kernel = normalized_weights
        
        return output

    return custom_forward
# def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
#     return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
