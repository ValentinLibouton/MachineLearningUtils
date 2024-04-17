import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import randint
import random
from pathlib import Path
import os


class ImagePredictionAnalyzer:
    def __init__(self, y_true, y_pred, preds_probs, class_names, directory, img_type='*.jpg'):
        """
        Initializes the analyzer with prediction results and directory information to manage prediction analysis.

        Parameters:
        - y_true (np.array): Array of true labels.
        - y_pred (np.array): Array of predicted labels.
        - preds_probs (np.array): Array of prediction probabilities from the model.
        - class_names (list of str): List of class names corresponding to labels.
        - directory (str): Path to the directory containing image files organized by class.
        - img_type (str, optional): Glob pattern to match image files. Default is '*.jpg'.

        The class is designed to analyze and visualize model prediction results effectively.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.pred_conf = preds_probs.max(axis=1)
        self.class_names = class_names
        self.directory = Path(directory)
        self.filepaths = [str(path) for path in self.directory.rglob(img_type)]
        self.pred_df = None


    def create_prediction_df(self):
        """
        Creates and returns a DataFrame containing prediction data including image paths, true labels, predicted labels,
        prediction confidence, and the correctness of the predictions.

        The DataFrame columns are:
        - 'img_path': Paths to each image file.
        - 'y_true': True labels for each prediction.
        - 'y_pred': Model's predicted labels.
        - 'pred_conf': Confidence level of the predictions.
        - 'y_true_classname': Class names corresponding to true labels.
        - 'y_pred_classname': Class names corresponding to predicted labels.
        - 'pred_correct': Boolean values indicating whether each prediction was correct.

        Returns:
        pd.DataFrame: A DataFrame populated with the prediction details.
        """
        self.pred_df = pd.DataFrame({
            "img_path": self.filepaths,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "pred_conf": self.pred_conf,
            "y_true_classname": [self.class_names[i] for i in self.y_true],
            "y_pred_classname": [self.class_names[i] for i in self.y_pred]
        })
        self.pred_df['pred_correct'] = self.pred_df['y_true'] == self.pred_df['y_pred']
        return self.pred_df


    def analyze_incorrect_predictions(self, top_n=100):
        """
        Analyzes and returns the top N incorrect predictions sorted by descending confidence.

        Parameters:
        - top_n (int, optional): Number of incorrect predictions to return. Default is 100.

        Returns:
        pd.DataFrame: A subset of the main DataFrame showing the top N incorrect predictions with highest confidence.
        """
        incorrect_df = self.pred_df[self.pred_df['pred_correct'] == False]
        return incorrect_df.sort_values('pred_conf', ascending=False).head(top_n)


    def visualize_predictions(self, dataframe, start_index=0, num_images=9):
        """
        Visualizes a selection of predictions using a DataFrame derived from prediction data.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing prediction data.
        - start_index (int, optional): Starting index in the DataFrame to begin visualization. Default is 0.
        - num_images (int, optional): Number of images to display. Default is 9.

        Displays images in a grid, annotating each image with the actual and predicted class names, and prediction confidence.
        Each image title is colored green for correct predictions and red for incorrect ones.
        """
        plt.figure(figsize=(15, 10))
        sample_df = dataframe[start_index:start_index+num_images]
        for i, row in enumerate(sample_df.itertuples()):
            img = plt.imread(row.img_path)
            plt.subplot(3, 3, i+1)
            plt.imshow(img)
            plt.title(f"actual: {row.y_true_classname}, pred: {row.y_pred_classname}\nprob: {row.pred_conf:.2f}")
            plt.axis(False)
        plt.show()



def prepare_image(file_path, img_shape=224, scale=True, channels=3):
    """
    Reads an image from filename, turns it into a tensor, reshapes it to
    (img_shape, img_shape, channels), and optionally scales the image values to [0, 1].

    Parameters:
    - file_path (str): Path to the target image.
    - img_shape (int): Desired size of the image sides.
    - scale (bool): Whether to scale pixel values to the range [0, 1].
    - channels (int): Number of color channels for the image.

    Returns:
    Tensor of the processed image.
    """
    try:
        # Read in the image
        img = tf.io.read_file(file_path)
        # Decode the read file into a tensor
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        # Resize the image
        img = tf.image.resize(img, size=[img_shape, img_shape])
        if scale:
            img = img / 255.0
        return img
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def predict_and_display_image(model, file_path, class_names):
    """
    Loads an image from a specified file path, makes a prediction using the provided model,
    and plots the image with the predicted class label as the title.

    Parameters:
    - model: A TensorFlow/Keras model that will be used to make predictions.
    - file_path (str): The file path to the image to be predicted.
    - class_names (list): A list of class names that correspond to the output layer of the model,
      used to map the prediction to a human-readable class name.

    Returns:
    None. This function directly shows a plot of the image with the predicted class label.
    """
    img = prepare_image(file_path)

    if img is None:
        print("Image preparation failed.")
        return
    try:
        # Make a prediction
        pred = model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class (binary or multi-class
        if len(pred[0]) > 1:
            pred_class = class_names[tf.argmax(pred[0])]
        else:
            pred_class = class_names[int(tf.round(pred[0]))]

        # Plot the image and predicted class
        plt.imshow(img)
        plt.title(f"Prediction: {pred_class}")
        plt.axis(False)
        plt.show()
    except Exception as e:
        print(f"An error occurred during prediction or visualization: {e}")


def predict_random_images(model, images, true_labels, classes, num_images=1, figsize=(15, 15)):
    """
    Selects multiple random images from a dataset, uses the provided model to make predictions on them,
    and displays the images with their predicted and true class labels.

    Parameters:
    - model: The trained model used for making predictions.
    - images: Array or list of images in the dataset.
    - true_labels: Array or list of true labels corresponding to the images.
    - classes: List of class names that correspond to the output layer of the model.
    - num_images: Number of random images to predict and display.

    Note: The function assumes that images are preprocessed appropriately for the model.
    """

    plt.figure(figsize=figsize)


    for img_num in range(num_images):
        # Set up random integer
        i = randint(0, len(images))

        # Create predictions and tagets
        target_image = images[i].reshape(1, *images[i].shape)
        pred_probs = model.predict(target_image)
        pred_label = classes[pred_probs.argmax()]
        true_label = classes[true_labels[i]]

        # Plot the image in a subplot
        plt.subplot(1, num_images, img_num + 1)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xticks([])  # Remove x ticks
        plt.yticks([])  # Remove y ticks

        # Change the color of the titles depending on if the prediction is right or
        # wrong
        color = "green" if pred_label == true_label else "red"

        # Add xlabel information (prediction/true label)
        plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                         100 * tf.reduce_max(pred_probs),
                                                         true_label),
                   color=color)  # set the color to green or red based on if prediction
        # is right or wrong
    plt.show()


def plot_decision_boundary(model, X, y):
    """
    Visualizes the decision boundary of a classification model on a 2D feature space. The function creates a mesh grid based on the feature values in X and uses the model to predict outcomes over this grid to plot the boundary.

    Parameters:
    - model: Trained classification model capable of making predictions.
    - X: Feature data, expected to be 2D for visualization.
    - y: True labels corresponding to X.

    Note: This function is designed for models with 2D feature input and binary or multiclass output.
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X value (we're going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # We have to reshape our prediction to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def predict_and_display_random_images_from_dir(model, dir_path, class_names, num_images, scale=False):
    """
    Displays a specified number of randomly selected images from a directory, predicts their classes using a provided model, and annotates them with the actual and predicted class names along with prediction confidence.

    This function is particularly useful for visually inspecting the performance of a classification model by comparing its predictions against true labels on a set of randomly selected images from a directory structured by class names.

    Parameters:
    - model (tf.keras.Model): The pre-trained model to use for making predictions.
    - dir_path (str): The path to the directory where the image classes are stored. Each class should be in its own subdirectory.
    - class_names (list of str): A list containing the names of the classes corresponding to the model's output layers.
    - num_images (int): The number of random images to display and predict.
    - scale (bool, optional): Whether to scale the image pixels to the range [0, 1]. Defaults to False.

    The images are expected to be organized in subdirectories within `dir_path`, where each subdirectory's name corresponds to the class names in `class_names`.

    Each subplot will display an image with a title that includes the actual class (from the directory name), the predicted class, and the highest prediction probability (confidence). Titles of correctly predicted images are shown in green, while incorrect predictions are shown in red.

    Note:
    - This function uses Matplotlib to create a figure containing subplots for each image. The figure's size is preset to be large enough to comfortably view `num_images`.
    - The function assumes that the images are suitable for the model in terms of size or other preprocessing requirements beyond scaling.

    Examples:
    >>> model = load_model('path_to_your_model')
    >>> directory_path = 'path_to_your_test_images'
    >>> class_names = ['class1', 'class2', 'class3']
    >>> predict_and_display_random_images_from_dir(model, directory_path, class_names, num_images=3, scale=True)

    This will load three random images, predict their classes using `model`, and display them with annotations.
    """
    plt.figure(figsize=(17,10))
    base_path = Path(dir_path)
    for img_num in range(num_images):
        class_name = random.choice(class_names)
        class_path = base_path / class_name
        filename = random.choice(os.listdir(class_path))
        filepath = str(class_path / filename)
        
        img = prepare_image(filepath, scale=scale)
        img_expanded = tf.expand_dims(img, axis=0)

        pred_prob = model.predict(img_expanded) # get prediction probabilities array
        pred_class = class_names[pred_prob.argmax()] # get highest prediction probability index and match it class_names list
        
        # Plot the image(s)
        plt.subplot(1, 3, img_num+1)
        # print(img)
        plt.imshow(img/225.)
        if class_name == pred_class: # if predicted class matches truth class, make text green
            title_color = "g"
        else:
            title_color = "r"
        plt.title(f"actual: {class_name}, pred: {pred_class}, prob: {pred_prob.max():.2f}", c=title_color)
        plt.axis(False)