import logging
from feat import Detector
import os

# Setup logging
logging.basicConfig(filename='run_detector(song).log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration
RAVDESS_path = "F:/RAVDESS_song/" # RAVDESS video path
tracked_path = "F:/raw_motion_song/" # result CSV path
start_actor_num = 1 # from Actor_01
end_actor_num = 25 # to Actor_24
isSong = True # RAVDESS song skip Actor 18

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


def main():
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
    for i in range(start_actor_num, end_actor_num):
        # Skip Actor 18 for RAVDESS song
        if i == 18 and isSong:
            continue
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
            
            # if the CSV file does not exists, then process the video with py-feat detector
            # otherwise skip this file since it has been processed
            if not os.path.exists(csv_path):
                run_feat_detector(detector, video_path, csv_path)
            else:
                logging.info(f"File {csv_path} already processed, skipping.")

if __name__ == '__main__':
    main()