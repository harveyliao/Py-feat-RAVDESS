import os
import wget

# Base URLs
url_song_template = "https://zenodo.org/records/1188976/files/Video_Song_Actor_{num_song}.zip"
url_speech_template = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_{num_speech}.zip"

# Folder to save the downloaded files
download_folder = r"F:\RAVDESS"

# Ensure the folder exists
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

def download_file(url, save_path):
    wget.download(url, save_path)

def main():
    # Download Video_Song_Actor files
    for i in range(1, 25):
        if i == 18:
            continue
        num_song = f"{i:02}"
        url_song = url_song_template.format(num_song=num_song)
        save_path_song = os.path.join(download_folder, f"Video_Song_Actor_{num_song}.zip")
        print(f"Downloading {url_song} to {save_path_song}...")
        download_file(url_song, save_path_song)
        print(f"Downloaded {url_song}.")

    # Download Video_Speech_Actor files
    for i in range(1, 25):
        num_speech = f"{i:02}"
        url_speech = url_speech_template.format(num_speech=num_speech)
        save_path_speech = os.path.join(download_folder, f"Video_Speech_Actor_{num_speech}.zip")
        print(f"Downloading {url_speech} to {save_path_speech}...")
        download_file(url_speech, save_path_speech)
        print(f"Downloaded {url_speech}.")

if __name__ == "__main__":
    main()
