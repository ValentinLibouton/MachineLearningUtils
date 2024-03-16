def download_and_extract_if_not_exist(filename, url):
    """
    Downloads and extracts a zip file from a specified URL if the zip file doesn't already exist.

    This function checks for the existence of a zip file by the given filename in the current directory.
    If the file does not exist, it downloads the zip file from the provided URL and extracts its contents
    to the current directory. If the file already exists, it skips the download and extraction steps
    and prints a message indicating that the file exists.

    Parameters:
    - filename (str): The name of the zip file to check for and to create. This is the local filename
      where the zip content will be saved if it needs to be downloaded.
    - url (str): The URL of the zip file to download. This is the source location from which the file
      will be downloaded if it does not already exist locally.

    Returns:
    None. The function directly downloads and extracts the file if needed, or prints a message if the file
    already exists.
    """
    if not os.path.exists(filename):
        !wget {url} -O {filename}

        print(f"Extraction de {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()
        print(f"{filename} a été téléchargé et extrait.")
    else:
        print(f"Le fichier {filename} existe déjà.")