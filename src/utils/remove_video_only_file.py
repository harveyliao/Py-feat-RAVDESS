import os
import glob
import logging

# Define the path to the main directory
RAVDESS_path = "F:/RAVDESS/"
delete_counter = 0
# Setup logging
logging.basicConfig(filename="remove_video_only_file.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Loop through each actor directory
for i in range(1, 25):
    # Format the folder name with leading zeros
    folder_name = f"Actor_{i:02}"
    full_path = os.path.join(RAVDESS_path, folder_name)
    
    # Find all .mp4 files that start with '02-' in the folder
    file_pattern = os.path.join(full_path, "02-*.mp4")
    files_to_delete = glob.glob(file_pattern)
    
    # Delete each file found
    for file in files_to_delete:
        os.remove(file)
        logging.info(f"Deleted {file}")
        delete_counter += 1

print(f"{delete_counter} files has been deleted")
logging.info(f"{delete_counter} files has been deleted")
