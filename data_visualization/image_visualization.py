import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import random


def display_random_images_from_class(dir_path, target_class, num_images=1):
    """
        Display random images from a specified class folder.

        Parameters:
            dir_path (str): The directory path containing class folders.
            target_class (str): The target class folder name.
            num_images (int): The number of random images to display. Default is 1.

        Returns:
            list or None: A list of image arrays if images are displayed successfully, else None.

        Raises:
            FileNotFoundError: If the target class folder doesn't exist.
            ValueError: If num_images is less than 1.
        """
    target_folder = os.path.join(dir_path, target_class)
    image_files = os.listdir(target_folder)
    image_list = []
    if not image_files:
        print(f"No images found in {target_folder}.")
        return

    if num_images > len(image_files):
        print(f"Requested {num_images} images, but only found {len(image_files)}. Displaying available images.")
        num_images = len(image_files)

    random_images = random.sample(image_files, num_images)
    try:
        for image_name in random_images:
            img_path = os.path.join(target_folder, image_name)
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.title(target_class)
            plt.axis("off")
            plt.show()
            print(f"Image shape: {img.shape}")
            image_list.append(img)
        return image_list

    except Exception as e:
        print(f"Error displaying image: {e}")
        return None
