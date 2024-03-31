import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf


def plot_original_and_augmented(original_images, augmented_images, index=None, figsize=(10, 5)):
    """
    Displays an original image and its augmented version side by side.

    Parameters:
    - original_images (numpy.ndarray or a tensor): Batch of original images.
    - augmented_images (numpy.ndarray or a tensor): Batch of augmented images corresponding to the original images.
    - index (int, optional): Index of the specific image to display. If None, a random index is chosen.
    - figsize (tuple, optional): Size of the figure as (width, height). Default is (10, 5).

    Returns:
    None. This function plots the images directly.
    """
    if index is None:
        index = random.randint(0, len(original_images) - 1)  # Choose a random index if not provided

    # Ensure the index is within the range of the images batch
    index = index % len(original_images)

    plt.figure(figsize=figsize)

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_images[index])
    plt.title("Original Image")
    plt.axis(False)  # 'False' is more conventional

    # Plot the augmented image
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_images[index])
    plt.title("Augmented Image")
    plt.axis(False)

    plt.show()
    

def apply_model_and_compare(img_path, model, figsize=(10, 5)):
    """
    Applies a given model to an image loaded from a path and displays the original and the model's output side by side.

    Parameters:
    - img_path (str): Path to the original image.
    - model: A TensorFlow/Keras model or any callable that takes an image as input and returns its transformed version.
    - figsize (tuple, optional): Size of the figure as (width, height). Default is (10, 5).

    Returns:
    None. This function plots the images directly.
    """

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The provided image path does not exist: {img_path}")

    # Load the image using mpimg
    original_img = mpimg.imread(img_path)
    
    # Apply the model to the image
    transformed_img = model(tf.expand_dims(original_img, axis=0))[0]

    if np.max(transformed_img) > 1.0:
        transformed_img = transformed_img / 255.  # Normalize pixel values if not already in [0, 1]

    transformed_img = tf.squeeze(transformed_img)

    plt.figure(figsize=figsize)

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img)
    plt.title("Model Output")
    plt.axis('off')

    plt.show()


def apply_model_and_compare_randomly(dir_path, target_class, model, num_images=1, figsize=(10, 5)):
    """
    Display random images from a specified class folder and their transformed version by a model side by side.

    Parameters:
    - dir_path (str): The directory path containing class folders.
    - target_class (str): The target class folder name.
    - model: A TensorFlow/Keras model or any callable that takes an image as input and returns its transformed version.
    - num_images (int): The number of random images to display and transform. Default is 1.
    - figsize (tuple, optional): Size of the figure as (width, height) for each pair of images. Default is (10, 5).

    Returns:
    None. This function plots the images directly.

    Raises:
    - FileNotFoundError: If the target class folder doesn't exist.
    - ValueError: If num_images is less than 1 or no images are found in the target class folder.
    """
    target_folder = os.path.join(dir_path, target_class)
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"Target class folder '{target_folder}' does not exist.")

    image_files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    if not image_files:
        raise ValueError(f"No images found in {target_folder}.")

    if num_images < 1:
        raise ValueError("num_images must be at least 1.")

    num_images = min(num_images, len(image_files))
    random_images = random.sample(image_files, num_images)

    for image_name in random_images:
        img_path = os.path.join(target_folder, image_name)
        original_img = mpimg.imread(img_path)
        
        transformed_img = model(tf.expand_dims(original_img, axis=0))[0]
        
        # Normalize
        if np.max(transformed_img) > 1:
            transformed_img = transformed_img /255.
        
        transformed_img = tf.squeeze(transformed_img)
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Original - {target_class}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(transformed_img)
        plt.title(f"Transformed - {target_class}")
        plt.axis("off")

        plt.show()