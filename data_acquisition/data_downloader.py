import os
import zipfile
import subprocess
def download_data(url, filename, extract=False):
    """
    Downloads a file from the specified URL if it doesn't already exist and optionally extracts it if it's a zip file.

    Parameters:
    - url (str): The URL of the file to download.
    - filename (str): The filename under which the file will be saved locally.
    - extract (bool, optional): Whether to extract the file if it's a zip file. Defaults to False.

    Returns:
    None
    """
    if not os.path.exists(filename):
        # Download the file using wget
        subprocess.run(["wget", "-nc", url, "-O", filename])
        print(f"{filename} has been downloaded.")

        # Extract the file if it's a zip and extraction is requested
        if extract and filename.endswith('.zip'):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall()
            print(f"{filename} has been extracted.")
    else:
        print(f"The file {filename} already exists.")