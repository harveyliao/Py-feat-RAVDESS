from feat import Detector
import os

# from Actor_01 to Actor_24
start_actor_num = 1
end_actor_num = 25

# set RAVDESS video path and output csv path
RAVDESS_path = "F:/RAVDESS/"
tracked_path = "F:/tracked/"

# load py-feat detector, using CUDA
# performance measures ~0.7s/frame
print("loading py-feat detector")
detector = Detector(device="cuda")

# Loop through each actor directory
for i in range(start_actor_num, end_actor_num):
    
    # Format the folder name with leading zeros
    folder_name = f"Actor_{i:02}"
    # Set the video input folder path and CSV output folder path
    actor_video_folder_path = os.path.join(RAVDESS_path, folder_name)
    actor_csv_folder_path = os.path.join(tracked_path, folder_name)
    # create folder for CSV output if the folder does not exist
    if not os.path.exists(actor_csv_folder_path):
        os.makedirs(actor_csv_folder_path)
        print(f"create folder {actor_csv_folder_path}")

    # traverse all videos in the actor folder
    for file_name in os.listdir(actor_video_folder_path):
        # complete the video path
        video_path = os.path.join(actor_video_folder_path, file_name)
        # extract RAVDESS coding, without file extension and path
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        # naming CSV file to the according video
        csv_path = os.path.join(actor_csv_folder_path, f"{video_basename}.csv")
        print(f"{csv_path=}")

        # if the CSV file exists, then skip this video, since it has already been processed
        if (os.path.exists(csv_path)):
            print("\tthis file has been processed, skip")
            continue
        
        # run py-feat detector
        print(f"Running detection for file {video_path}:")
        video_prediction = detector.detect_video(video_path)

        # save to CSV
        video_prediction.to_csv(csv_path)