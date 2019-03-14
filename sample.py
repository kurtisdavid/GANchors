import tensorflow as tf
import tensorflow_hub as hub

# Load BigGAN 256 module.
module = hub.Module('https://tfhub.dev/deepmind/biggan-256/2')

# Sample random noise (z) and ImageNet label (y) inputs.
batch_size = 8
truncation = 0.5  # scalar truncation value in [0.02, 1.0]
z = truncation * tf.random.truncated_normal([batch_size, 140])  # noise sample
y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [8, 256, 256, 3] and range [-1, 1].
samples = module(dict(y=y, z=z, truncation=truncation))

print(samples.shape)
