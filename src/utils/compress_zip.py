import os
import zipfile

def zip_folder(base_path, start, end):
    for i in range(start, end + 1):
        folder_name = f"Actor_{i:02d}"
        archive_name = f"{folder_name}.zip"
        folder_path = os.path.join(base_path, folder_name)
        archive_path = os.path.join(base_path, archive_name)

        # Create a ZipFile object in write mode
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the directory
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    # Create the path to file
                    file_path = os.path.join(root, file)
                    # Create the archive name
                    in_zip_path = os.path.relpath(file_path, start=os.path.dirname(folder_path))
                    # Add file to zip
                    zipf.write(file_path, in_zip_path)
            print(f"Compressed {folder_name}")

if __name__ == "__main__":
    # Set the base path where the folders are located
    base_path = "F:\\tracked"
    # Call the function with the range of folders you want to compress
    zip_folder(base_path, 1, 24)
