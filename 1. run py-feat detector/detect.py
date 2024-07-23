import logging
from feat import Detector
import os

# Setup logging
logging.basicConfig(filename='video_processing_song.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration
RAVDESS_path = "F:/RAVDESS_song/" # RAVDESS video path
tracked_path = "F:/tracked_song/" # result CSV path
start_actor_num = 1 # from Actor_01
end_actor_num = 25 # to Actor_24
isSong = True # RAVDESS song skip Actor 18

# Load detector
logging.info("Loading py-feat detector")
face_model = 'img2pose'
landmark_model='mobilenet' 
facepose_model='img2pose-c'
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

def process_video(video_path, csv_path):
    #TODO: wirte comment
    """
    input:
    output:
    
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
        video_prediction.to_csv(csv_path)
        logging.info(f"Output saved to {csv_path}")
    except Exception as e:
        logging.error(f"Error processing file {video_path}: {e}")

# Main processing loop
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
    if not os.path.exists(actor_csv_folder_path):
        os.makedirs(actor_csv_folder_path)
        logging.info(f"Created folder {actor_csv_folder_path}")
    
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
            process_video(video_path, csv_path)
        else:
            logging.info(f"File {csv_path} already processed, skipping.")
