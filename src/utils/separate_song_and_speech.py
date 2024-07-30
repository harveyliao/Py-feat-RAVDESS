import os
import shutil
import logging

# Define the path to the main directory
source_path = "F:/smoothed_video/Landmark"
song_path = "F:/smoothed_video_song/Landmark"
speech_path = "F:/smoothed_video_speech/Landmark"

# Setup logging
logging.basicConfig(filename="separate_song_and_speech.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def create_folder_if_not_exist(folder_path):
    """create the folder at folder_path if it does not exist, with logging"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"Created folder {folder_path}")

def is_song_coding(coding_str: str) -> bool:
    """Return True if the RAVDESS 'Channel' identifier is '02' (Song), otherwise False.
    
    Argument:
    :param coding_str: Factor-level coding of RAVDESS filename, e.g., '01-01-03-02-02-01-09'
    :return: True if the 'Channel' identifier is '02', otherwise False
    """
    return coding_str.split('-')[1] == '02'


def is_speech_coding(coding_str: str) -> bool:
    """Return True if the RAVDESS 'Channel' identifier is '01' (Speech), otherwise False.
    
    Argument:
    :param coding_str: Factor-level coding of RAVDESS filename, e.g., '01-01-03-02-02-01-09'
    :return: True if the 'Channel' identifier is '01', otherwise False
    """
    return coding_str.split('-')[1] == '01'


# Loop through each actor directory
for i in range(1, 25):
    # Format the folder name with leading zeros
    folder_name = f"Actor_{i:02}"
    full_source_path = os.path.join(source_path, folder_name)
    full_song_path = os.path.join(song_path, folder_name)
    full_speech_path = os.path.join(speech_path, folder_name)

    create_folder_if_not_exist(full_song_path)
    create_folder_if_not_exist(full_speech_path)

    # print(full_source_path, full_song_path, full_speech_path)
    for file_name in os.listdir(full_source_path):
        file_path = os.path.join(full_source_path, file_name)
        
        video_basename = os.path.splitext(os.path.basename(file_name))[0]
        # print(video_basename)
        if is_song_coding(video_basename):
            shutil.copy(file_path, full_song_path)
            logging.info(f"{file_path} is copied to Song directory")
        elif is_speech_coding(video_basename):
            shutil.copy(file_path, full_speech_path)
            logging.info(f"{file_path} is copied to Speech directory")

        
    

