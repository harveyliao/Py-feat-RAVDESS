import logging
import os
import io
import imageio.v2 as imageio
from feat.utils.io import read_feat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Setup logging
logging.basicConfig(filename='draw_landmark.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration
csv_path = "F:/smoothed/"
video_path = "F:/video/Landmark/"
start_actor_num = 1
end_actor_num = 25
isSong = False
num_processes = 10

# Overlay video settings
DPI = 100

def generate_landmark_video_from_csv(args):
    """Generate an landmark video from a CSV feat dataframe.

    Landmark video includes landmarks, facebox, and head pose. Eliminating translational motion

    :param args: A tuple containing two elements: 
        - smoothed_csv_path: path of the smoothed CSV file
        - landmark_video_path: path where the AU video is saved to

    """
    smoothed_csv_path, landmark_video_path = args
    
    # if the target file exists, then skip it since it has been processed. Othrewise proceed
    if os.path.exists(landmark_video_path):
        logging.info(f"File {landmark_video_path} already processed, skipping.")
        return

    video_prediction = read_feat(smoothed_csv_path)
    logging.info(f"generating overlay for each frame, file: {smoothed_csv_path}")
    
    figs = video_prediction.plot_detections(faces='landmarks', 
                                            faceboxes=True, 
                                            muscles=False, 
                                            poses=True, 
                                            gazes=False, 
                                            add_titles=False, 
                                            au_barplot=False, 
                                            emotion_barplot=False, 
                                            plot_original_image=False)

    writer = imageio.get_writer(landmark_video_path, fps=30, codec='libx264', format='FFMPEG', macro_block_size=None)

    logging.info("generating video")
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=DPI)
        buf.seek(0)
        image = imageio.imread(buf)
        writer.append_data(image)
        buf.close()
        plt.close(fig)

    writer.close()
    logging.info(f'Video saved as {landmark_video_path}')

def main():
    tasks = [] # multiprocessing pool

    for i in range(start_actor_num, end_actor_num):
        if i == 18 and isSong:
            continue
        folder_name = f"Actor_{i:02}" 
        smoothed_csv_folder_path = os.path.join(csv_path, folder_name)
        landmark_video_folder_path = os.path.join(video_path, folder_name)

        if not os.path.exists(landmark_video_folder_path):
            os.makedirs(landmark_video_folder_path)
            logging.info(f"Created folder {landmark_video_folder_path}")
        
        for file_name in os.listdir(smoothed_csv_folder_path):
            smoothed_csv_path = os.path.join(smoothed_csv_folder_path, file_name)
            video_basename = os.path.splitext(os.path.basename(smoothed_csv_path))[0]
            landmark_video_path = os.path.join(landmark_video_folder_path, f"{video_basename}.mp4")
            
            tasks.append((smoothed_csv_path, landmark_video_path))
    
    with Pool(processes=num_processes) as pool:
        pool.map(generate_landmark_video_from_csv, tasks)

if __name__ == '__main__':
    main()
