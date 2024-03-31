from tensorflow import reduce_mean


def global_average_pooling2d(inputs):
    """
    Applies Global Average Pooling to the spatial dimensions (height and width) of a 4D tensor.

    Args:
    inputs (Tensor): A 4D shape tensor (batch_size, height, width, channels).

    Returns:
    Tensor: A 2D shape tensor (batch_size, channels) after applying Global Average Pooling.
    """
    return reduce_mean(inputs, axis=[1, 2])
