from pathlib import Path
from datetime import datetime
import tensorflow as tf


def create_tensorboard_callback(dir_name, experiment_name, update_freq='epoch'):
    """
    Creates and returns a TensorBoard callback.

    :param dir_name: Parent directory for the TensorBoard logs.
    :param experiment_name: Name of the experiment, used to name the subdirectory for logs.
    :param update_freq: Frequency at which logs are written, 'batch', 'epoch' or an integer. Defaults to 'epoch'.
    :return: Configured instance of the TensorBoard callback for the specified experiment.
    """
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(dir_name) / experiment_name / current_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), update_freq=update_freq)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback
