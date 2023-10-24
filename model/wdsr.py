# import tensorflow_addons as tfa
import tensorflow
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras import backend as K

from model.common import normalize, denormalize, pixel_shuffle


def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    # main branch
    m = Conv2DWeightNorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = Conv2DWeightNorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = Conv2DWeightNorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="wdsr")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = Conv2DWeightNorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = Conv2DWeightNorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = Conv2DWeightNorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = Conv2DWeightNorm(int(num_filters * linear), 1, padding='same')(x)
    x = Conv2DWeightNorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

class Conv2DWeightNorm(tensorflow.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', activation=None, **kwargs):
        super(Conv2DWeightNorm, self).__init__()
        
        self.conv = Conv2D(filters, kernel_size, padding=padding, activation=None, **kwargs)
        self.conv.kernel_initializer = VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        self.data_init = False

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.conv.build(input_shape)
        
        # Create the scaling and bias variables for weight normalization
        self.g = self.add_weight(shape=(input_dim,), initializer="ones", name="g", trainable=True)
        self.b = self.add_weight(shape=(input_dim,), initializer="zeros", name="b", trainable=True)
        self.initialized = True

    def call(self, inputs):
        if not self.initialized:
            self.build(inputs.shape)

        # Calculate the scaling factor
        self.conv.kernel = tf.nn.l2_normalize(self.conv.kernel, axis=(0, 1, 2)) * self.g

        # Apply convolution
        x = self.conv(inputs)

        # Apply bias
        x = x + self.b

        if self.conv.activation is not None:
            x = self.conv.activation(x)

        return x
# def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
#     return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
