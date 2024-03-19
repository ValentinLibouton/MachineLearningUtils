import os
import zipfile
import subprocess
import tarfile
try:
    import magic
except ImportError:
    magic = None
try:
    import py7zr
except ImportError:
    py7zr = None

def extract_zip_file(filename, destination=None):
    """
    Extracts a ZIP file to a specified destination directory.

    This function opens a ZIP file specified by 'filename' and extracts its contents to the directory specified by 'destination'. If 'destination' is not provided, the contents are extracted to the current working directory. The function prints messages indicating the start and completion of the extraction process.

    Parameters:
    - filename (str): The path to the ZIP file to be extracted.
    - destination (str, optional): The directory to which the contents of the ZIP file should be extracted. Defaults to None, which results in extraction to the current working directory.
    """
    print(f"Extracting {filename} as ZIP...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(path=destination)
    print(f"{filename} has been extracted to {destination if destination else 'current directory'}.")

def extract_tar_file(filename, destination=None):
    """
    Extracts a TAR.GZ file to a specified destination directory.

    Opens a TAR.GZ file specified by 'filename' and extracts its contents to the directory specified by 'destination'. If 'destination' is not provided, the contents are extracted to the current working directory. Messages are printed to indicate the start and completion of the extraction process.

    Parameters:
    - filename (str): The path to the TAR.GZ file to be extracted.
    - destination (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    """
    print(f"Extracting {filename} as TAR.GZ...")
    with tarfile.open(filename, 'r:gz') as tar_ref:
        tar_ref.extractall(path=destination)
    print(f"{filename} has been extracted to {destination if destination else 'current directory'}.")

def extract_7z_file(filename, destination=None):
    """
    Extracts a 7z file to a specified destination directory, if the py7zr module is available.

    Attempts to open a 7z file specified by 'filename' and extract its contents to 'destination'. If 'destination' is not provided, the contents are extracted to the current working directory. If the py7zr module is not installed, prints a message indicating that the module is required for extracting 7z files.

    Parameters:
    - filename (str): The path to the 7z file to be extracted.
    - destination (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    """
    if py7zr:
        print(f"Extracting {filename} as 7z...")
        with py7zr.SevenZipFile(filename, mode='r') as zip_ref:
            zip_ref.extractall(path=destination)
        print(f"{filename} has been extracted to {destination if destination else 'current directory'}.")
    else:
        print("py7zr module not available. Unable to extract .7z files.")

def extract_archive_file(filename, destination=None):
    """
    Determines the type of an archive file using the magic module and extracts it to a specified destination.

    Uses the magic module to identify the MIME type of 'filename' and calls the appropriate function to extract its contents. Supports ZIP, TAR.GZ, TAR, BZIP2, and 7Z formats. If the magic module is not available, prints an error message.

    Parameters:
    - filename (str): The path to the archive file to be extracted.
    - destination (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    """
    if magic:
        mime_type = magic.from_file(filename, mime=True)
        if mime_type == 'application/zip':
            extract_zip_file(filename, destination)
        elif mime_type in ['application/gzip', 'application/x-gzip']:
            extract_tar_file(filename, destination, mode='r:gz')
        elif mime_type == 'application/x-tar':
            extract_tar_file(filename, destination, mode='r:')
        elif mime_type == 'application/x-bzip2':
            extract_tar_file(filename, destination, mode='r:bz2')
        elif mime_type == 'application/x-7z-compressed':
            extract_7z_file(filename, destination)
        else:
            print(f"Archive type {mime_type} not supported")
    else:
        print("magic module not available. Unable to determine archive type. "
              "Please install python-magic to enable this feature.")


def download_data(url, filename, extract=False, destination=None, remove_archive=False):
    """
    Downloads a file from a specified URL and optionally extracts it.

    Downloads a file from 'url' to 'filename'. If 'extract' is True, attempts to extract the file using 'extract_archive_file'. If 'remove_archive' is True, deletes the archive file after extraction. Prints messages to indicate the status of download and extraction.

    Parameters:
    - url (str): The URL from which to download the file.
    - filename (str): The local filename to save the downloaded file.
    - extract (bool, optional): Whether to extract the downloaded file. Defaults to False.
    - destination (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    - remove_archive (bool, optional): Whether to remove the archive file after extraction. Defaults to False.
    """
    if not os.path.exists(filename):
        # Download the file using wget
        subprocess.run(["wget", "-nc", url, "-O", filename])
        print(f"{filename} has been downloaded.")
    else:
        print(f"The file {filename} already exists.")

    if extract:
        extract_archive_file(filename, destination)

        # Remove the archive file if requested
        if remove_archive:
            os.remove(filename)
            print(f"{filename} has been removed after extraction.")
