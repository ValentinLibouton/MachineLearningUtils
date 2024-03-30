import tensorflow as tf


def sigmoid(x):
  return 1 / (1 + tf.math.exp(-x))


def relu(x):
  return tf.math.maximum(0, x)


def softmax(x):
  return tf.nn.softmax(x)


def tanh(x):
    return tf.math.tanh(x)


def leaky_relu(x, alpha=0.01):
    return tf.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0):
    return tf.where(x > 0, x, alpha * (tf.math.exp(x) - 1))


def swish(x):
    return x * sigmoid(x)
