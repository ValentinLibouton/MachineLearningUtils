import tensorflow as tf


def activate_all_gpus():
    """
    Attempts to activate GPU memory growth for TensorFlow to enable efficient GPU utilization.

    This function iterates through all available GPUs and attempts to enable memory growth on them.
    Enabling memory growth allows TensorFlow to allocate GPU memory as needed rather than allocating
    the full memory upfront. This can be beneficial when sharing the GPU with other applications or
    when running multiple models simultaneously.

    If GPUs are found and memory growth is successfully enabled, a confirmation message is printed.
    If TensorFlow encounters a RuntimeError during this process (commonly due to memory issues),
    the error is caught and printed. If no GPUs are found, a message indicating that TensorFlow will
    use the CPU is printed.

    Note: This function should be called before initializing any TensorFlow models or operations
    that require GPU usage.

    Raises:
        RuntimeError: If TensorFlow encounters a memory-related issue while enabling memory growth
                      on the GPUs. The error is printed but not raised further, allowing the program
                      to continue (potentially in CPU mode).

    Prints:
        A message indicating the successful activation of GPU memory growth, an error message if
        a RuntimeError is encountered, or a message indicating fallback to CPU if no GPUs are found.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU not found. TensorFlow will use the CPU.")

def activate_gpu():
    """
    Attempts to activate GPU memory growth for TensorFlow to enable efficient GPU utilization.

    This function iterates through all available GPUs and attempts to enable memory growth on them.
    Enabling memory growth allows TensorFlow to allocate GPU memory as needed rather than allocating
    the full memory upfront. This can be beneficial when sharing the GPU with other applications or
    when running multiple models simultaneously.

    If GPUs are found and memory growth is successfully enabled, a confirmation message is printed.
    If TensorFlow encounters a RuntimeError during this process (commonly due to memory issues),
    the error is caught and printed. If no GPUs are found, a message indicating that TensorFlow will
    use the CPU is printed.

    Note: This function should be called before initializing any TensorFlow models or operations
    that require GPU usage.

    Raises:
        RuntimeError: If TensorFlow encounters a memory-related issue while enabling memory growth
                      on the GPUs. The error is printed but not raised further, allowing the program
                      to continue (potentially in CPU mode).

    Prints:
        A message indicating the successful activation of GPU memory growth, an error message if
        a RuntimeError is encountered, or a message indicating fallback to CPU if no GPUs are found.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    else:
        print("GPU not found. TensorFlow will use the CPU.")

