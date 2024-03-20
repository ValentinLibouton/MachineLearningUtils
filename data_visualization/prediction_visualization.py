import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import randint


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


def predict_random_image(model, images, true_labels, classes):
    """
    Selects a random image from a dataset, uses the provided model to make a prediction on it,
    and displays the image with the predicted and true class labels.

    Parameters:
    - model: The trained model used for making predictions.
    - images: Array or list of images in the dataset.
    - true_labels: Array or list of true labels corresponding to the images.
    - classes: List of class names that correspond to the output layer of the model.

    Note: The function assumes that images are preprocessed appropriately for the model.
    """

    # Set up random integer
    i = randint(0, len(images))

    # Create predictions and tagets
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    # Plot the image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Change the color of the titles depending on if the prediction is right or
    # wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # Add xlabel information (prediction/true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100 * tf.reduce_max(pred_probs),
                                                     true_label),
               color=color)  # set the color to green or red based on if prediction
    # is right or wrong


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
