import os

def create_temp_folders(base_folder, subfolders):
    """
    Create temporary folders within a base folder.

    Parameters:
    - base_folder (str): Path to the base folder.
    - subfolders (list): A list of subfolders to be created within the base folder.
    """
    # Create the base folder if it doesn't exist
    os.makedirs(base_folder, exist_ok=True)
    temp_paths={}
    # Create temporary folders within the base folder
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)
        temp_paths[subfolder]=subfolder_path
    return temp_paths

