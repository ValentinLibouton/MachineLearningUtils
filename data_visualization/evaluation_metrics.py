from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(5, 5), text_size=15, norm=False, savefig=False):
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
