from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def make_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, classes: list = None, figsize: tuple = (5, 5), text_size: int = 15, norm: bool = False, savefig: bool = False, xlabels_rotation: int = 0):
    """
    Generates and plots a confusion matrix from true labels and predicted labels.

    Parameters:
    y_true (np.ndarray): True labels of the data.
    y_pred (np.ndarray): Predicted labels by the classifier.
    classes (list, optional): List of class names for the axis labels. If not provided, integer labels are used.
    figsize (tuple, optional): A tuple representing the figure size. Defaults to (5, 5).
    text_size (int, optional): Font size of the labels in the plot. Defaults to 15.
    norm (bool, optional): If True, the confusion matrix will be normalized. Defaults to False.
    savefig (bool, optional): If True, the figure will be saved as 'confusion_matrix.png' in the current directory. Defaults to False.
    xlabels_rotation (int, optional): Degrees to rotate the x-axis labels. Defaults to 0 for horizontal labels.

    Raises:
    ValueError: If `y_true` and `y_pred` are not of the same length.

    Returns:
    None: This function does not return anything but generates a matplotlib plot and optionally saves it as an image.
    """
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"The length of y_true and y_pred must be the same. len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")
    
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize our confusion matrix
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Plot x-labels vertically
    plt.xticks(rotation=xlabels_rotation, fontsize=text_size)
    plt.yticks(fontsize=text_size)


    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    if savefig:
        fig.savefig("confusion_matrix.png", dpi=300)


def evaluate_classification_metrics(y_true, y_pred):
    """
    Evaluates key metrics for a binary classification model including accuracy, precision, recall, and F1 score.

    This function provides a comprehensive assessment of a binary classifier's performance by computing its accuracy, precision, recall, and F1 score. These metrics are crucial for understanding the model's ability to correctly identify positive and negative classes, balance between precision and recall, and achieve a harmonic mean of the two with the F1 score. The function is suitable for a wide range of binary classification tasks.

    Args:
    y_true (array-like): True labels of the data, expected to be a 1D array of binary values.
    y_pred (array-like): Predicted labels as determined by the classifier, expected to be a 1D array of binary values.

    Returns:
    dict: A dictionary containing the calculated metrics: accuracy, precision, recall, and F1 score. Each metric is provided as a floating-point value representing the model's performance in that specific area.

    Example:
    results = evaluate_binary_classification_metrics(y_true=[0, 1, 1, 0], y_pred=[1, 1, 1, 0])
    print(results)  # Output might be: {'accuracy': 75.0, 'precision': 0.66, 'recall': 1.0, 'f1': 0.8}
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results
