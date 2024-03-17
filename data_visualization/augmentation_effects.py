import random
import matplotlib.pyplot as plt



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