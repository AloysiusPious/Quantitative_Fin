import os
import shutil
import glob

def remove_directory(directory):
    """Remove directory if it exists"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Directory '{directory}' removed successfully.")
    else:
        print(f"Directory '{directory}' not found.")

if __name__ == "__main__":
    # List of directories to remove
    directories_to_remove = ["Reports", "Charts", "Summary", "Master"]

    # Iterate over each directory and remove it if it exists
    for directory in directories_to_remove:
        for dir_path in glob.glob(f'*{directory}*'):
            remove_directory(dir_path)