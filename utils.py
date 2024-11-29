from pathlib import Path
import hashlib

def calculate_file_hash(file_path):
    """
    Calculate the hash of a file.

    :param file_path: Path to the file to be hashed.
    :return: Hexadecimal hash of the file.
    :raises FileNotFoundError: If the file does not exist or is not a file.
    """

    path = Path(file_path)

    # Check if the file exists and is a file
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist or is not a file.")
    
    hash_func = hashlib.new('sha256')

    # Read the file in chunks
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):  
            hash_func.update(chunk)

    return hash_func.hexdigest()
