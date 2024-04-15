import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



class ClassificationReportProcessor:
    def __init__(self, y_true, y_pred, class_names=None):
        """
        Initializes the processor with true and predicted labels to generate a classification report.
        
        Parameters:
        y_true (array-like): Actual true labels of the dataset.
        y_pred (array-like): Labels predicted by the classifier.
        class_names (list of str, optional): Custom names for the classes. If not provided, numeric labels are used.
        """
        self.__y_true = y_true
        self.__y_pred = y_pred
        self.__class_names = class_names if class_names else sorted(set(y_true) | set(y_pred), key=lambda x: x)
        self.__report = classification_report(y_true, y_pred, output_dict=True)
        self.__unique_labels = sorted(set(self.__y_true) | set(self.__y_pred))
        self.__totals_list = ['accuracy', 'macro avg', 'weighted avg']
        self.__validate_and_replace_class_names()

            
    def __validate_and_replace_class_names(self):
        """
        Validates if the custom class names match the unique labels from the dataset and replaces numeric labels in the report if valid.
        """
        if self.__class_names:
            if len(self.__class_names) != len(self.__unique_labels):
                warnings.warn("The length of new_keys does not match the number of keys in original_dict. Returning original dictionary.")
                return
            # Map from old class labels to new class names
            label_to_name = {str(old): new for old, new in zip(self.__unique_labels, self.__class_names)}
            self.__report = {label_to_name.get(key, key): value for key, value in self.__report.items()}
                

    def get_class_report(self, class_name):
        """
        Retrieve the classification metrics for a given class.

        Parameters:
        class_name (str): The class name to retrieve the report for.

        Returns:
        dict: A dictionary of classification metrics for the specified class, or a message if the class is not found.
        """
        return self.__report.get(str(class_name), "Class not found")
    

    @property
    def class_names(self):
        """
        Property that returns the class names used in the classification report, excluding any aggregate statistics entries.

        Returns:
        list: A list of class names as strings.
        """
        return [key for key in self.__report.keys() if key not in self.__totals_list]


    @property
    def f1_scores(self):
        """
        Returns a dictionary of f1-scores for all classes.

        Returns:
        dict: Dictionary with class names as keys and f1-scores as values.
        """
        return {class_name: info['f1-score'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}
    
    
    @property
    def precisions(self):
        """
        Returns a dictionary of precisions for all classes.

        Returns:
        dict: Dictionary with class names as keys and precisions as values.
        """
        return {class_name: info['precision'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}
    

    @property
    def recalls(self):
        """
        Returns a dictionary of recalls for all classes.

        Returns:
        dict: Dictionary with class names as keys and recalls as values.
        """
        return {class_name: info['recall'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}


    @property
    def supports(self):
        """
        Returns a dictionary of supports for all classes.

        Returns:
        dict: Dictionary with class names as keys and support numbers as values.
        """
        return {class_name: info['support'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}
    

    @property
    def average_precision(self):
        """
        Calculate the weighted average precision across all classes.

        Returns:
        float: Weighted average precision.
        """
        return self.__report['weighted avg']['precision']


    @property
    def average_recall(self):
        """
        Calculate the weighted average recall across all classes.

        Returns:
        float: Weighted average recall.
        """
        return self.__report['weighted avg']['recall']


    @property
    def average_f1_score(self):
        """
        Calculate the weighted average f1-score across all classes.

        Returns:
        float: Weighted average f1-score.
        """
        return self.__report['weighted avg']['f1-score']


    @property
    def total_support(self):
        """
        Sum the support across all classes, which corresponds to the total number of samples.

        Returns:
        int: Total number of samples (support).
        """
        return sum(class_info['support'] for class_name, class_info in self.__report.items() if class_name not in self.__totals_list)
    

    @property
    def confusion_matrix(self):
        """
        Property that returns the confusion matrix as a numpy array.

        Returns:
        np.ndarray: The confusion matrix of the true versus predicted labels.
        """
        return confusion_matrix(self.__y_true, self.__y_pred)


    def make_confusion_matrix(self, figsize=(5, 5), text_size=15, norm=False, savefig=False, xlabels_rotation=0):
        """
        Generates and plots a confusion matrix from true labels and predicted labels.

        Parameters:
        figsize (tuple, optional): A tuple representing the figure size. Defaults to (5, 5).
        text_size (int, optional): Font size of the labels in the plot. Defaults to 15.
        norm (bool, optional): If True, the confusion matrix will be normalized. Defaults to False.
        savefig (bool, optional): If True, the figure will be saved as 'confusion_matrix.png' in the current directory. Defaults to False.
        xlabels_rotation (int, optional): Degrees to rotate the x-axis labels. Defaults to 0 for horizontal labels.
        """
        cm = self.confusion_matrix
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if norm else cm
        labels = self.__class_names

        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        ax.set(title="Confusion Matrix",
               xlabel="Predicted Label",
               ylabel="True Label",
               xticks=np.arange(len(labels)),
               yticks=np.arange(len(labels)),
               xticklabels=labels,
               yticklabels=labels)

        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        plt.xticks(rotation=xlabels_rotation, fontsize=text_size)
        plt.yticks(fontsize=text_size)

        threshold = (cm.max() + cm.min()) / 2
        for i, j in itertools.product(range(len(labels)), range(len(labels))):
            plt.text(j, i, f"{cm_norm[i, j]:.0f}" if norm else f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

        if savefig:
            fig.savefig("confusion_matrix.png", dpi=300)
        plt.show()

    
    @property
    def confusion_matrix_to_dataframe(self):
        """
        Property that returns the confusion matrix as a pandas DataFrame, which can be used for plotting or analysis.

        Returns:
        DataFrame: The confusion matrix of the true versus predicted labels, indexed by class names.
        """
        return pd.DataFrame(self.confusion_matrix, index=self.__class_names, columns=self.__class_names)

    
    @property
    def specificity(self):
        """
        Calculate the specificity for each class, which is the true negative rate.

        Returns:
        dict: Dictionary of specificity for each class.
        """
        cm = self.confusion_matrix
        specificity_dict = {}
        total_instances = cm.sum()
        
        for idx, class_name in enumerate(self.__class_names):
            # Calculate TN: Sum everything but the row and column for the current class
            tn = total_instances - (cm[idx, :].sum() + cm[:, idx].sum() - cm[idx, idx])
            # Calculate FP: Sum the column for the current class minus the diagonal (true positives)
            fp = cm[:, idx].sum() - cm[idx, idx]
            # Calculate specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_dict[class_name] = specificity
        
        return specificity_dict

    
    @property
    def specificity_to_dataframe(self):
        """
        Converts the specificity scores for each class into a pandas DataFrame for easier analysis and visualization.
    
        This property retrieves the specificity values calculated for each class, then organizes these values into a DataFrame with two columns: 'Class' and 'Specificity'. Each row in the DataFrame corresponds to a class and its associated specificity score, facilitating clear and structured presentation of the data.
    
        Returns:
        DataFrame: A DataFrame where each row represents a class and its corresponding specificity score, with columns labeled 'Class' and 'Specificity'.
        """
        return pd.DataFrame(list(self.specificity.items()), columns=['Class', 'Specificity'])
    
    
    @property
    def report_to_dataframe(self):
        """
        Converts the classification report to a pandas DataFrame for easier analysis and visualization.
        
        Returns:
        DataFrame: A DataFrame containing precision, recall, f1-score, specificity, and support for each class.
        """
        return pd.DataFrame({
            'class_name': self.class_names,
            'precision': self.precisions.values(),
            'recall': self.recalls.values(),
            'specificity': self.specificity.values(),
            'f1-score': self.f1_scores.values(),
            'support': self.supports.values()
        })



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
