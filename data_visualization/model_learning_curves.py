import matplotlib.pyplot as plt


def plot_loss_curves(history):
    """
    Plots the loss and accuracy curves for training and validation.

    This function takes a TensorFlow History object as input and generates two plots: one for the loss and the other for the accuracy. It automatically distinguishes between training and validation metrics based on the keys in the History object, accommodating different metric names.

    Parameters:
    - history (History): The History object obtained from the `fit` method of a TensorFlow/Keras model. It contains record of loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

    Returns:
    None. This function directly plots the loss and accuracy curves using Matplotlib, splitting the loss and accuracy into separate plots for better clarity.
    """
    loss_=None
    val_loss_=None
    accuracy_=None
    val_accuracy_=None
    for elem in history.history.keys():
        if "val" in elem and "loss" in elem:
            val_loss_ = elem
        elif "val" in elem and "accuracy" in elem:
            val_accuracy_ = elem
        elif "accuracy" in elem:
            accuracy_ = elem
        elif "loss" in elem:
            loss_ = elem
            
    loss = history.history[loss_]
    val_loss = history.history[val_loss_]
    
    accuracy = history.history[accuracy_]
    val_accuracy = history.history[val_accuracy_]
    
    epochs = range(len(history.history[loss_])) # how many epochs did we run for?
    
    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    
    plt.figure() # Splitting into 2 plot
    
    # Plot accuracy
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()


def plot_loss_curves_v2(history):
    """
    Plots separate loss curves for training and validation metrics.
    """
    metrics = list(history.history.keys())
    loss_keys = [s for s in metrics if "loss" in s and "val" not in s]
    val_loss_keys = [s for s in metrics if "loss" in s and "val" in s]
    acc_keys = [s for s in metrics if "acc" in s and "val" not in s]
    val_acc_keys = [s for s in metrics if "acc" in s and "val" in s]
    
    # Plot loss curves
    for loss_key, val_loss_key in zip(loss_keys, val_loss_keys):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history[loss_key], label="training_loss")
        plt.plot(history.history[val_loss_key], label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    # Plot accuracy curves
    for acc_key, val_acc_key in zip(acc_keys, val_acc_keys):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history[acc_key], label="training_accuracy")
        plt.plot(history.history[val_acc_key], label="val_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()


def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    """

    # Dynamic detection of metric names
    def find_metric_names(history):
        loss_key, val_loss_key, accuracy_key, val_accuracy_key = None, None, None, None
        for key in history.history.keys():
            if "val" in key and "loss" in key:
                val_loss_key = key
            elif "val" in key and "acc" in key:  # To handle 'acc' and 'accuracy
                val_accuracy_key = key
            elif "acc" in key and "val" not in key:  # To handle 'acc' and 'accuracy
                accuracy_key = key
            elif "loss" in key and "val" not in key:
                loss_key = key
        return loss_key, val_loss_key, accuracy_key, val_accuracy_key

    # Obtaining metric names for the original history
    loss_key, val_loss_key, accuracy_key, val_accuracy_key = find_metric_names(original_history)

    # Get original history measurements
    acc_values = original_history.history[accuracy_key]
    loss_values = original_history.history[loss_key]

    val_acc_values = original_history.history[val_accuracy_key]
    val_loss_values = original_history.history[val_loss_key]

    # Combining the original history with the new history
    total_acc = acc_values + new_history.history[accuracy_key]
    total_loss = loss_values + new_history.history[loss_key]

    total_val_acc = val_acc_values + new_history.history[val_accuracy_key]
    total_val_loss = val_loss_values + new_history.history[val_loss_key]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    