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


def create_model_checkpoint(dir_name, experiment_name, monitor="val_loss", save_best_only=True, verbose=0):
    """
    Create a callback for a TensorFlow Keras model that saves the model as a checkpoint.

    The callback saves the model at a specified directory and under a specific experiment name. The model
    is saved based on the monitoring of a certain metric (default is 'val_loss') and can be configured to save
    only the best performing model.

    Args:
        dir_name (str): The directory name where the model checkpoint will be saved.
        experiment_name (str): The name of the experiment under which the model checkpoint will be saved.
        monitor (str, optional): The metric name to monitor. Defaults to 'val_loss'.
        save_best_only (bool, optional): Whether to save only the model that achieved the best
            performance on the monitored metric. Defaults to True.
        verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            Defaults to 0.

    Returns:
        ModelCheckpoint: A TensorFlow Keras callback that can be used in model training to save the model.

    Example:
        checkpoint_callback = create_model_checkpoint("models", "exp1")
        model.fit(x_train, y_train, callbacks=[checkpoint_callback])
    """
    filepath = Path(dir_name) / experiment_name
    return tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                              monitor=monitor,
                                              verbose=verbose,
                                              save_best_only=save_best_only)