import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


# def psnr(x1, x2):
#     if x1.shape.as_list() != x2.shape.as_list():
#         x1 = tf.image.resize_with_pad(x1, x2.shape[1], x2.shape[2])
#     return tf.image.psnr(x1, x2, max_val=255)

# Ensure both images have the same shape and data type
def psnr(original,reconstructed):
    if original.shape != reconstructed.shape:
        original_height, original_width, _ = original.shape
        reconstructed = np.array(Image.fromarray(reconstructed).resize((original_width, original_height)))
    # Calculate the Mean Squared Error (MSE) between the two images
    mse = np.mean((original - reconstructed) ** 2)
    
    # If the MSE is very close to zero, return a high PSNR value (e.g., infinity)
    if mse == 0:
        return float('inf')
    
    # Calculate the PSNR using the formula: PSNR = 10 * log10(MAX^2 / MSE)
    max_pixel_value = 255  # Assuming pixel values range from 0 to 255 (8-bit images)
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
    return psnr
    


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


