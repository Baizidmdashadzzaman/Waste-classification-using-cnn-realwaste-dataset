import os


def rename_images_in_folder(folder_path, prefix):
    """
    Renames all files in a folder by adding a specified prefix.

    Args:
        folder_path (str): The path to the folder containing the files.
        prefix (str): The prefix to add to each file name.
    """
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder at '{folder_path}' was not found.")
        return

    print(f"Starting to rename files in '{folder_path}'...")

    # Get a list of all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the full old and new file paths
        old_file_path = os.path.join(folder_path, filename)
        new_file_name = f"{prefix}_{filename}"
        new_file_path = os.path.join(folder_path, new_file_name)

        # Check if it's a file before renaming
        if os.path.isfile(old_file_path):
            try:
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"  - Renamed '{filename}' to '{new_file_name}'")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

    print("\nFile renaming completed.")


if __name__ == "__main__":
    # Specify the folder you want to rename files in
    # This example assumes the script is in the same directory as your 'train' folder
    target_folder = 'diversionnet/Validation/Paper'

    # Specify the prefix you want to add
    new_prefix = 'vali'

    # Run the function
    rename_images_in_folder(target_folder, new_prefix)
