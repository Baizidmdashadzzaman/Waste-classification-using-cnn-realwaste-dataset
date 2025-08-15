import os
import shutil
import random

def create_train_test_split(source_dir, output_dir, split_ratio=0.8):
    """
    Splits a dataset into training and testing folders with a specified ratio.

    Args:
        source_dir (str): The path to the source directory containing subdirectories for each class.
        output_dir (str): The path to the output directory where 'train' and 'test' folders will be created.
        split_ratio (float): The ratio of images to be used for the training set (e.g., 0.8 for 80%).
    """
    # Create the main output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define paths for the train and test directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    # Create the train and test directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Iterate through each class subdirectory in the source directory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)

        # Skip if it's not a directory
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing class: {class_name}")

        # Create subdirectories for the current class in the train and test folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Get a list of all image files in the current class directory
        all_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Shuffle the list of images randomly
        random.shuffle(all_images)

        # Calculate the split point
        split_point = int(len(all_images) * split_ratio)

        # Split the images into training and testing sets
        train_images = all_images[:split_point]
        test_images = all_images[split_point:]

        # Copy the training images to the train directory
        for img_path in train_images:
            shutil.copy(img_path, os.path.join(train_dir, class_name))

        # Copy the testing images to the test directory
        for img_path in test_images:
            shutil.copy(img_path, os.path.join(test_dir, class_name))

        print(f"  - Moved {len(train_images)} images to train/{class_name}")
        print(f"  - Moved {len(test_images)} images to test/{class_name}")

if __name__ == "__main__":
    # Define the source and output directories
    source_directory = 'realwaste'  # The folder with your dataset
    output_directory = 'realwaste_split' # The new folder for train/test splits

    # Run the function to create the split
    create_train_test_split(source_directory, output_directory, split_ratio=0.8)

    print("\nDataset split completed successfully!")
    print(f"Check the '{output_directory}' folder for your 'train' and 'test' sets.")
