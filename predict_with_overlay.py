from feat import Detector # py-feat
import imageio.v2 as imageio # to create video from frames
import io

detector = Detector(device="cuda")

video_path = input("Enter the RAVDESS video path: ")

# Constants for video settings
VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720  # Desired resolution
DPI = 100  # Resolution in dots per inch

print("running py-feat detector")
# run detector
video_prediction = detector.detect_video(video_path)

print("generating overlay for each frame")
# generate overlay for each frame
figs = video_prediction.plot_detections(faces='landmarks', 
                                          faceboxes=True, 
                                          muscles=False, 
                                          poses=True, 
                                          gazes=False, 
                                          add_titles=False, 
                                          au_barplot=False, 
                                          emotion_barplot=False, 
                                          plot_original_image=True
                                          )

# Create a video writer object, specifying the output file, codec, and framerate
output_filename = f'{video_path[:-4]}_tracked.mp4'
writer = imageio.get_writer(output_filename, fps=30, codec='libx264', format='FFMPEG', macro_block_size=None)

print("generating video")
# Loop through each figure in the list, adjust size, save it to a buffer, and append it to the video
for fig in figs:
    # Set the figure size to correspond to the desired pixel dimensions
    fig.set_size_inches(VIDEO_WIDTH / DPI, VIDEO_HEIGHT / DPI)
    
    # Use a buffer to save the figure
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI)
    buf.seek(0)
    
    # Read the image from the buffer and add it to the video
    image = imageio.imread(buf)
    writer.append_data(image)
    
    # Close the buffer
    buf.close()

# Close the writer to finalize the video
writer.close()

print(f'Video saved as {output_filename}')
