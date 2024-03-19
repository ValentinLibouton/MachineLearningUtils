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

def extract_zip_file(file_path, extract_to=None):
    """
    Extracts a ZIP file to a specified extract_to directory.

    This function opens a ZIP file specified by 'file_path' and extracts its contents to the directory specified by 'extract_to'. If 'extract_to' is not provided, the contents are extracted to the current working directory. The function prints messages indicating the start and completion of the extraction process.

    Parameters:
    - file_path (str): The path to the ZIP file to be extracted.
    - extract_to (str, optional): The directory to which the contents of the ZIP file should be extracted. Defaults to None, which results in extraction to the current working directory.
    """
    print(f"Extracting {file_path} as ZIP...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_to)
    print(f"{file_path} has been extracted to {extract_to if extract_to else 'current directory'}.")

def extract_tar_file(file_path, extract_to=None):
    """
    Extracts a TAR.GZ file to a specified extract_to directory.

    Opens a TAR.GZ file specified by 'file_path' and extracts its contents to the directory specified by 'extract_to'. If 'extract_to' is not provided, the contents are extracted to the current working directory. Messages are printed to indicate the start and completion of the extraction process.

    Parameters:
    - file_path (str): The path to the TAR.GZ file to be extracted.
    - extract_to (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    """
    print(f"Extracting {file_path} as TAR.GZ...")
    with tarfile.open(file_path, 'r:gz') as tar_ref:
        tar_ref.extractall(path=extract_to)
    print(f"{file_path} has been extracted to {extract_to if extract_to else 'current directory'}.")

def extract_7z_file(file_path, extract_to=None):
    """
    Extracts a 7z file to a specified extract_to directory, if the py7zr module is available.

    Attempts to open a 7z file specified by 'file_path' and extract its contents to 'extract_to'. If 'extract_to' is not provided, the contents are extracted to the current working directory. If the py7zr module is not installed, prints a message indicating that the module is required for extracting 7z files.

    Parameters:
    - file_path (str): The path to the 7z file to be extracted.
    - extract_to (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    """
    if py7zr:
        print(f"Extracting {file_path} as 7z...")
        with py7zr.SevenZipFile(file_path, mode='r') as zip_ref:
            zip_ref.extractall(path=extract_to)
        print(f"{file_path} has been extracted to {extract_to if extract_to else 'current directory'}.")
    else:
        print("py7zr module not available. Unable to extract .7z files.")

def extract_archive_file(file_path, extract_to=None):
    """
    Determines the type of an archive file using the magic module and extracts it to a specified extract_to.

    Uses the magic module to identify the MIME type of 'file_path' and calls the appropriate function to extract its contents. Supports ZIP, TAR.GZ, TAR, BZIP2, and 7Z formats. If the magic module is not available, prints an error message.

    Parameters:
    - file_path (str): The path to the archive file to be extracted.
    - extract_to (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    """
    if magic:
        mime_type = magic.from_file(file_path, mime=True)
        if mime_type == 'application/zip':
            extract_zip_file(file_path, extract_to)
        elif mime_type in ['application/gzip', 'application/x-gzip']:
            extract_tar_file(file_path, extract_to, mode='r:gz')
        elif mime_type == 'application/x-tar':
            extract_tar_file(file_path, extract_to, mode='r:')
        elif mime_type == 'application/x-bzip2':
            extract_tar_file(file_path, extract_to, mode='r:bz2')
        elif mime_type == 'application/x-7z-compressed':
            extract_7z_file(file_path, extract_to)
        else:
            print(f"Archive type {mime_type} not supported")
    else:
        print("magic module not available. Unable to determine archive type. "
              "Please install python-magic to enable this feature.")


def download_data(url, file_path, extract=False, extract_to=None, remove_archive=False):
    """
    Downloads a file from a specified URL and optionally extracts it.

    Downloads a file from 'url' to 'file_path'. If 'extract' is True, attempts to extract the file using 'extract_archive_file'. If 'remove_archive' is True, deletes the archive file after extraction. Prints messages to indicate the status of download and extraction.

    Parameters:
    - url (str): The URL from which to download the file.
    - file_path (str): The local file_path to save the downloaded file.
    - extract (bool, optional): Whether to extract the downloaded file. Defaults to False.
    - extract_to (str, optional): The directory where the contents should be extracted. Defaults to the current working directory.
    - remove_archive (bool, optional): Whether to remove the archive file after extraction. Defaults to False.
    """
    if not os.path.exists(file_path):
        # Download the file using wget
        subprocess.run(["wget", "-nc", url, "-O", file_path])
        print(f"{file_path} has been downloaded.")
    else:
        print(f"The file {file_path} already exists.")

    if extract:
        extract_archive_file(file_path, extract_to)

        # Remove the archive file if requested
        if remove_archive:
            os.remove(file_path)
            print(f"{file_path} has been removed after extraction.")
