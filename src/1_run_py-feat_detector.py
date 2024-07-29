import logging
from feat import Detector
import os

RAVDESS_path = "F:/RAVDESS/" # RAVDESS video path
tracked_path = "F:/raw_motion/"  # result CSV path
start_actor_num = 1 # from Actor_01
end_actor_num = 25 # to Actor_24

# User selection for configuration
print("Select configuration to run:")
print("1. Speech only")
print("2. Song only")
print("3. All (default)")
choice = input("Enter the number of your choice: ")

if choice == "1":
    logging_filename = 'run_detector(speech_only).log'
    is_song = False
    is_speech_and_song = False
elif choice == "2":
    logging_filename = 'run_detector(song_only).log'
    is_song = True  # RAVDESS song skip Actor 18
    is_speech_and_song = False
else:
    logging_filename = 'run_detector(all).log'
    is_speech_and_song = True
    is_song = False

# Setup logging
logging.basicConfig(filename=logging_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load detector
face_model = 'img2pose'
landmark_model='mobilenet' 
facepose_model='img2pose-c'

def run_feat_detector(detector, video_path, csv_path):
    """run feat.detector.detect_video() with the video and save the result to a CSV

    Arguments:
    :param detector:    py-feat detector
    :param video_path:  path of the to be detected video
    :param csv_path:    location where the result is saved, this path should include .csv extension

    """
    # if the target file exists, then skip it since it has been processed. Othrewise proceed
    if os.path.exists(csv_path):
        logging.info(f"File {csv_path} already processed, skipping.")
        return

    try:
        # logging.info(f"Running detection for file {video_path}")
        print(f"Running detection for file {video_path}")
        video_prediction = detector.detect_video(video_path, 
                                                skip_frames=None, 
                                                output_size=(720, 1280), 
                                                batch_size=5, 
                                                num_workers=0, 
                                                pin_memory=False, 
                                                face_detection_threshold=0.83, 
                                                face_identity_threshold=0.8
                                                )
        save_feat_prediction_to_csv(video_prediction, csv_path)
    except Exception as e:
        logging.error(f"Error processing file {video_path}: {e}")

def save_feat_prediction_to_csv(feat_prediction, csv_path):
    """save feat prediction to csv files, with logging
    
    Arguments:
    :param feat_prediction: A py-feat prediction dataframe to be saved
    :param csv_path:        location where the result is saved, this path should include .csv extension
    
    """
    try:
        feat_prediction.to_csv(csv_path)
        logging.info(f"Output saved to {csv_path}")
    except Exception as e:
        logging.error(f"Error saving file to {csv_path}: {e}")

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

def filter_tasks(tasks: list, is_speech_and_song: bool, is_song: bool) -> list:
    """Filter the pending tasks list based on two booleans, returning the list of filtered tasks.
    
    There are three cases based on the combination of the two boolean parameters:
    1. All: is_speech_and_song = True, is_song = Any
       - Includes both speech and song tasks, skips Actor 18 in Song.
    2. Speech Only: is_speech_and_song = False, is_song = False
       - Includes only speech tasks.
    3. Song Only: is_speech_and_song = False, is_song = True
       - Includes only song tasks, skips Actor 18 in Song.
    
    Arguments:
    :param tasks: List of tasks, where each task is a list with the RAVDESS filename at index 3.
    :param is_speech_and_song: True if intended to run both speech and song, otherwise False.
    :param is_song: True if intended to run song, otherwise False.
    :return: List of filtered tasks, each task is a 3-tuple: (detector, video_path, csv_path)
    """
    def get_actor(filename: str) -> int:
        """Extract and return the actor identifier from the RAVDESS filename."""
        return int(filename.split('-')[-1])

    def should_include_task(task, is_speech_and_song, is_song) -> bool:
        """checks if a task should be included based on conditions"""
        video_basename = task[3]
        actor_id = get_actor(video_basename)
        
        if is_speech_and_song: # case 1
            return not (actor_id == 18 and is_song_coding(video_basename))
        elif not is_speech_and_song and not is_song: # case 2
            return is_speech_coding(video_basename)
        elif not is_speech_and_song and is_song: # case 3
            return actor_id != 18 and is_song_coding(video_basename)
        return False

    results = [task[:3] for task in tasks if should_include_task(task, is_speech_and_song, is_song)]
    
    return results

def main():
    tasks = []

    # load feat Detector
    logging.info("Loading py-feat detector")
    detector = Detector(face_model=face_model, 
                        landmark_model=landmark_model, 
                        au_model='xgb', 
                        emotion_model='resmasknet', 
                        facepose_model=facepose_model,
                        identity_model='facenet', 
                        device='cuda', 
                        n_jobs=1, 
                        verbose=False,
                        )
    logging.info("py-feat detector loaded")

    # identify all the tasks
    for i in range(start_actor_num, end_actor_num):
        # Format the folder name with leading zeros
        folder_name = f"Actor_{i:02}" 
        # Set the video input folder path and CSV output folder path
        actor_video_folder_path = os.path.join(RAVDESS_path, folder_name)
        actor_csv_folder_path = os.path.join(tracked_path, folder_name)

        # create folder for CSV output if the folder does not exist
        create_folder_if_not_exist(actor_csv_folder_path)
        
        # traverse all videos in the actor folder
        for file_name in os.listdir(actor_video_folder_path):
            # complete the video path
            video_path = os.path.join(actor_video_folder_path, file_name)
            # extract RAVDESS coding, without file extension and path
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            # naming CSV file to the according video
            csv_path = os.path.join(actor_csv_folder_path, f"{video_basename}.csv")
            
            tasks.append((detector, video_path, csv_path, video_basename))

    tasks = filter_tasks(tasks, is_speech_and_song, is_song) # now tasks is a list of 3-tuple
    logging.info("Task filtering complete")
    
    # Iterate over the tasks and call run_feat_detector
    logging.info("Start running feat detector")
    for params in tasks:
        run_feat_detector(*params)

if __name__ == '__main__':
    main()