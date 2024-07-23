import os
import glob

# Define the path to the main directory
RAVDESS_path = "F:/RAVDESS/"
isSong = False

# Loop through each actor directory
for i in range(1, 25):
    if i == 18 and isSong:
        continue
    # Format the folder name with leading zeros
    folder_name = f"Actor_{i:02}"
    full_path = os.path.join(RAVDESS_path, folder_name)
    
    # Find all .mp4 files that start with '02-' in the folder
    file_pattern = os.path.join(full_path, "02-*.mp4")
    files_to_delete = glob.glob(file_pattern)
    
    # Delete each file found
    for file in files_to_delete:
        os.remove(file)
        print(f"Deleted {file}")
