import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import inspect
import functools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def apply_defaults(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        defaults = {
            'figsize': self.DEFAULT_FIGSIZE,
            'fontsize': self.DEFAULT_FONTSIZE,
            'savefig': self.DEFAULT_SAVEFIG,
            'xlabels_rotation': self.DEFAULT_XLABELS_ROTATION,
            'sort_ascending': self.DEFAULT_SORT_ASCENDING
        }
        # Get the signature of the function
        sig = inspect.signature(func)
        # For each default key, check if it exists in the function's parameters
        for key, default in defaults.items():
            if key in sig.parameters:
                # Set default only if not already specified in kwargs
                kwargs.setdefault(key, default)
        
        return func(self, *args, **kwargs)
    return wrapper

class ClassificationReportProcessor:
    # Class-level default attributes
    DEFAULT_FIGSIZE = (12, 12)
    DEFAULT_FONTSIZE = 12
    DEFAULT_NORMALIZE = False
    DEFAULT_SAVEFIG = False
    DEFAULT_XLABELS_ROTATION = 0
    DEFAULT_SORT_ASCENDING = True
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
    def class_names(self) -> list:
        """
        Property that returns the class names used in the classification report, excluding any aggregate statistics entries.

        Returns:
        list: A list of class names as strings.
        """
        return [key for key in self.__report.keys() if key not in self.__totals_list]


    @property
    def f1_scores(self) -> pd.DataFrame:
        """
        Returns a DataFrame of f1-scores for all classes.

        Returns:
        DataFrame: DataFrame with class names as indices and f1-scores as values.
        """
        f1_scores_dict = {class_name: info['f1-score'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}
        return pd.DataFrame(list(f1_scores_dict.items()), columns=['class', 'f1-scores']).set_index('class')
    
    
    @property
    def precisions(self) -> pd.DataFrame:
        """
        Returns a DataFrame of precision values for each class evaluated by the classifier.

        Precision measures the accuracy of positive predictions. It is the ratio of correctly predicted positive observations to the total predicted positives. This property constructs a DataFrame where each row corresponds to a class and its precision value, highlighting the classifier's accuracy in predicting each class as a positive instance.

        Returns:
        DataFrame: DataFrame with class names as indices and precision values as column data. This setup allows for straightforward analysis or visualization of the precision metric across different classes.
        """
        precisions_dict = {class_name: info['precision'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}
        return pd.DataFrame(list(precisions_dict.items()), columns=['class', 'precisions']).set_index('class')
    

    @property
    def recalls(self) -> pd.DataFrame:
        """
        Returns a DataFrame of recall values for each class evaluated by the classifier.

        Recall, also known as sensitivity, measures the ability of the classifier to find all the relevant cases within a class. This property generates a DataFrame where each row represents a class with its associated recall value, effectively showing how well the classifier can identify each class without missing cases.

        Returns:
        DataFrame: DataFrame with class names as indices and recall values as column data, facilitating easy access and manipulation of the recall metrics for further analysis or visualization.
        """
        recalls_dict = {class_name: info['recall'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}
        return pd.DataFrame(list(recalls_dict.items()), columns=['class', 'recalls']).set_index('class')


    @property
    def specificity(self) -> pd.DataFrame:
        """
        Calculate the specificity for each class, which is the true negative rate.

        Returns:
        DataFrame: A DataFrame containing the specificity for each class with columns 'Class' and 'Specificity'.
        """
        cm = self.confusion_matrix
        total_instances = cm.sum()
        class_specificities = []

        for idx, class_name in enumerate(self.__class_names):
            # Calculate TN: Sum everything but the row and column for the current class
            tn = total_instances - (cm[idx, :].sum() + cm[:, idx].sum() - cm[idx, idx])
            # Calculate FP: Sum the column for the current class minus the diagonal (true positives)
            fp = cm[:, idx].sum() - cm[idx, idx]
            # Calculate specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            class_specificities.append((class_name, specificity))

        return pd.DataFrame(class_specificities, columns=['class', 'specificity'])


    @property
    def supports(self) -> pd.DataFrame:
        """
        Returns a DataFrame of support counts for all classes.

        This method returns the support counts (the number of instances for each class) from the classification report,
        structured into a DataFrame for easy analysis and visualization.

        Returns:
        DataFrame: A DataFrame with 'class' as one column and 'supports' as another column, where 'class' is set as the DataFrame's index.
        """
        supports_dict = {class_name: info['support'] for class_name, info in self.__report.items() if class_name not in self.__totals_list}
        return pd.DataFrame(list(supports_dict.items()), columns=['class', 'supports']).set_index('class')


    @property
    def report_to_dataframe(self) -> pd.DataFrame:
        """
        Converts the classification report to a pandas DataFrame for easier analysis and visualization, with all metric columns in lowercase.

        This method concatenates several DataFrames, each containing a different metric for all classes,
        into a single DataFrame for a comprehensive view of classifier performance.

        Returns:
        DataFrame: A DataFrame containing precision, recall, f1-score, specificity, and support for each class,
        with class names as a column.
        """
        metric_df = pd.concat([
            self.class_names,
            self.precisions,
            self.recalls,
            self.specificity,
            self.f1_scores,
            self.supports
        ], axis=1)
        return metric_df.reset_index().rename(columns={'index': 'class_name'})


    ##########################################################################
    # Averages & Totals
    ##########################################################################

    @property
    def average_precision(self) -> float:
        """
        Calculate the weighted average precision across all classes.

        Returns:
        float: Weighted average precision.
        """
        return self.__report['weighted avg']['precision']


    @property
    def average_recall(self) -> float:
        """
        Calculate the weighted average recall across all classes.

        Returns:
        float: Weighted average recall.
        """
        return self.__report['weighted avg']['recall']


    @property
    def average_f1_score(self) -> float:
        """
        Calculate the weighted average f1-score across all classes.

        Returns:
        float: Weighted average f1-score.
        """
        return self.__report['weighted avg']['f1-score']


    @property
    def total_support(self) -> int:
        """
        Sum the support across all classes, which corresponds to the total number of samples.

        Returns:
        int: Total number of samples (support).
        """
        return sum(class_info['support'] for class_name, class_info in self.__report.items() if class_name not in self.__totals_list)

    ##########################################################################
    # Confusion Matrix & Plotting
    ##########################################################################

    @property
    def confusion_matrix(self) -> np.ndarray:
        """
        Property that returns the confusion matrix as a numpy array.

        Returns:
        np.ndarray: The confusion matrix of the true versus predicted labels.
        """
        return confusion_matrix(self.__y_true, self.__y_pred)

    @apply_defaults
    def make_confusion_matrix(self, figsize, fontsize, normalize, savefig, xlabels_rotation):
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
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
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
        plt.xticks(rotation=xlabels_rotation, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        threshold = (cm.max() + cm.min()) / 2
        for i, j in itertools.product(range(len(labels)), range(len(labels))):
            plt.text(j, i, f"{cm_norm[i, j]:.0f}" if normalize else f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=fontsize)

        if savefig:
            fig.savefig("confusion_matrix.png", dpi=300)
        plt.show()

    
    @property
    def confusion_matrix_to_dataframe(self) -> pd.DataFrame:
        """
        Property that returns the confusion matrix as a pandas DataFrame, which can be used for plotting or analysis.

        Returns:
        DataFrame: The confusion matrix of the true versus predicted labels, indexed by class names.
        """
        return pd.DataFrame(self.confusion_matrix, index=self.__class_names, columns=self.__class_names)


    @apply_defaults
    def plot_f1_scores(self, figsize, fontsize, savefig, sort_ascending):
        """
        Plots the F1-scores for each class in a horizontal bar chart.

        This method visualizes the F1-scores for all classes handled by the processor,
        providing a clear and intuitive graphical representation of model performance across different classes.

        Parameters:
        figsize: A tuple defining the dimensions of the figure (width, height). The size impacts how the plot is displayed.
        fontsize: The font size for all text elements in the plot, including axis labels and titles. This setting helps in making the plot more readable.
        savefig: A boolean that, if True, saves the plot to a file named 'f1_scores.png' in the current working directory. This option allows for easy preservation and sharing of the plot.

        The plot inverts the y-axis to display higher F1 scores at the top, enhancing the visual impact and readability of performance differences among classes.
        """
        f1_scores = self.f1_scores.sort_values(by='f1-scores', ascending=sort_ascending)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(f1_scores.index, f1_scores['f1-scores'])
        ax.set_xlabel("F1 Score", fontsize=fontsize)
        ax.set_title("F1 Scores for different classes", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if savefig:
            plt.savefig("f1_scores.png", dpi=300)
        plt.show()


    @apply_defaults
    def plot_precisions(self, figsize, fontsize, savefig, sort_ascending):
        """
        Plots the precisions for each class in a horizontal bar chart.

        Precision measures the accuracy of the positive predictions. This method visualizes the precision scores
        for all classes handled by the processor, offering a clear and intuitive graphical representation of
        the model's ability to correctly identify positive instances across different classes.

        Parameters:
        figsize: A tuple defining the dimensions of the figure (width, height).
        fontsize: The font size for all text elements in the plot, including axis labels and titles.
        savefig: A boolean that determines whether to save the plot to a file named 'precisions.png'.

        The plot inverses the y-axis to display higher scores at the top for better visual interpretation.
        """
        precisions = self.precisions.sort_values(by='precisions', ascending=sort_ascending)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(precisions.index, precisions['precisions'])
        ax.set_xlabel("Precision", fontsize=fontsize)
        ax.set_title("Precisions for different classes", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if savefig:
            plt.savefig("precisions.png", dpi=300)
        plt.show()


    @apply_defaults
    def plot_recalls(self, figsize, fontsize, savefig, sort_ascending):
        """
        Plots the recalls for each class in a horizontal bar chart.

        Recall measures the ability of the model to identify all relevant instances within each class. This method
        visualizes the recall scores for all classes, providing a clear and intuitive graphical representation of
        the model's sensitivity across different classes.

        Parameters:
        figsize: A tuple defining the dimensions of the figure (width, height).
        fontsize: The font size for all text elements in the plot, including axis labels and titles.
        savefig: A boolean that determines whether to save the plot to a file named 'recalls.png'.

        The plot inverses the y-axis to ensure higher values are at the top, aiding in quick visual assessment.
        """
        recalls = self.recalls.sort_values(by='recalls', ascending=sort_ascending)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(recalls.index, recalls['recalls'])
        ax.set_xlabel("Recall", fontsize=fontsize)
        ax.set_title("Recalls for different classes",fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if savefig:
            plt.savefig("recalls.png", dpi=300)
        plt.show()


    @apply_defaults
    def plot_specificity(self, figsize, fontsize, savefig, sort_ascending):
        """
        Plots the specificity for each class in a horizontal bar chart.

        Specificity measures the true negative rate for each class. This method visualizes the specificity scores
        for all classes handled by the processor, providing a clear and intuitive graphical representation of
        the model's performance in identifying true negatives across different classes.

        Parameters:
        figsize: A tuple defining the dimensions of the figure (width, height).
        fontsize: The font size for all text elements in the plot, including axis labels and titles.
        savefig: A boolean that determines whether to save the plot to a file named 'specificity.png'.

        The y-axis is inverted to display higher values at the top for easier comparison and better visual clarity.
        """
        specificity = self.specificity.sort_values(by='specificity', ascending=sort_ascending)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(specificity.index, specificity['specificity'])
        ax.set_xlabel("Specificity", fontsize=fontsize)
        ax.set_title("Specificity for different classes", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if savefig:
            plt.savefig("specificity.png", dpi=300)
        plt.show()


    @apply_defaults
    def plot_supports(self, figsize, fontsize, savefig, sort_ascending):
        """
        Plots the support counts for each class in a horizontal bar chart.

        Support counts indicate the number of samples for each class in the dataset. This method visualizes
        the support counts for all classes, providing a clear and intuitive graphical representation of the
        sample distribution across different classes.

        Parameters:
        figsize: A tuple defining the dimensions of the figure (width, height).
        fontsize: The font size for all text elements in the plot, including axis labels and titles.
        savefig: A boolean that determines whether to save the plot to a file named 'supports.png'.

        The y-axis is inverted to highlight classes with more samples at the top, enhancing readability and comparison.
        """
        supports = self.supports.sort_values(by='supports', ascending=sort_ascending)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(supports.index, supports['supports'])
        ax.set_xlabel("Supports", fontsize=fontsize)
        ax.set_title("Supports for different classes", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if savefig:
            plt.savefig("supports.png", dpi=300)
        plt.show()



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
