import tensorflow as tf


def sigmoid(x):
  return 1 / (1 + tf.math.exp(-x))


  def relu(x):
  return tf.math.maximum(0, x)