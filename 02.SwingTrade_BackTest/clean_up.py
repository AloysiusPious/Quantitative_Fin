import os
import shutil
import glob

def remove_directory():
    directories_to_remove = ["Reports", "Charts", "Summary", "Master", "Cvs_Data"]
    #directories_to_remove = ["Reports", "Charts", "Summary", "Master"]
    for directory in directories_to_remove:
        for dir_path in glob.glob(f'*{directory}*'):
            """Remove directory if it exists"""
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' removed successfully.")
            else:
                print(f"Directory '{dir_path}' not found.")
remove_directory()