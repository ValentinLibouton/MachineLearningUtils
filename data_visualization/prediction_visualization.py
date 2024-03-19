import matplotlib.pyplot as plt
import tensorflow as tf


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
            img = img/255.0
        return img
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def predict_and_visualize(model, file_path, class_names):
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
        # Get the predicted class
        pred_class = class_names[int(tf.round(pred))]
        # Plot the image and predicted class
        plt.imshow(img)
        plt.title(f"Prediction: {pred_class}")
        plt.axis(False)
        plt.show()
    except Exception as e:
        print(f"An error occurred during prediction or visualization: {e}")