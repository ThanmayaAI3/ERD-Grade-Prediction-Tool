import os
import shutil


def move_png_files(src_folder, train_folder, val_folder):
    # Ensure train and val folders exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # Loop through all files in the source folder
    for filename in os.listdir(src_folder):
        if filename.endswith(".png"):
            # Get the file number from the filename (e.g., 001 from 001.png)
            file_num = int(filename.split('.')[0])

            # Construct full file paths
            src_file = os.path.join(src_folder, filename)

            # Move files based on their number
            if 1 <= file_num <= 112:
                dst_file = os.path.join(train_folder, filename)
                shutil.move(src_file, dst_file)
                print(f"Moved {filename} to train folder")
            elif 113 <= file_num <= 140:
                dst_file = os.path.join(val_folder, filename)
                shutil.move(src_file, dst_file)
                print(f"Moved {filename} to val folder")


# Example usage
src_folder = 'Collection_labels_by_a_student/Collection_1'
train_folder = 'dataset/images/train'
val_folder = 'dataset/images/val'

move_png_files(src_folder, train_folder, val_folder)
