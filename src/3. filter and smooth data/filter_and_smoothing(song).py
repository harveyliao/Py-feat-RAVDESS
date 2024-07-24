import logging
import os
import pandas as pd
import numpy as np

from feat.utils.io import read_feat # load data from csv to FEX
from scipy.signal import butter, filtfilt # lowpass butterworth filter
from scipy.signal import savgol_filter # Savitzky-Golay filter

from multiprocessing import Pool

# Setup logging
logging.basicConfig(filename='filter_and_smooth_data(song).log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration
tracked_path = "F:/tracked_song/" # RAVDESS video path
smoothed_path = "F:/smoothed_song/" # result CSV path
start_actor_num = 1 # from Actor_01
end_actor_num = 25 # to Actor_24
isSong = True # To skip Actor 18 in RAVDESS song
num_processes = 10 # adjust this according to host machine performance

# columns to be filtered
columns_to_filter = [
	'FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight', 
	'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 
    'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 
    'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 
    'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 
    'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 
    'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 
    'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 
    'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'y_10', 
    'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 
    'y_21', 'y_22', 'y_23', 'y_24', 'y_25', 'y_26', 'y_27', 'y_28', 'y_29', 'y_30', 
    'y_31', 'y_32', 'y_33', 'y_34', 'y_35', 'y_36', 'y_37', 'y_38', 'y_39', 'y_40', 
    'y_41', 'y_42', 'y_43', 'y_44', 'y_45', 'y_46', 'y_47', 'y_48', 'y_49', 'y_50', 
    'y_51', 'y_52', 'y_53', 'y_54', 'y_55', 'y_56', 'y_57', 'y_58', 'y_59', 'y_60', 
    'y_61', 'y_62', 'y_63', 'y_64', 'y_65', 'y_66', 'y_67',
	'Pitch', 'Roll', 'Yaw', 
	'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 'AU12', 
    'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43'
]

def apply_lowpass_filter(column, cutoff_freq, sampling_freq, order=5):
    """apply lowpass butterworth filter to one column of data
    
    Arguments:
    :param column:          one column of all data to be filtered, in dataframe
    :param cutoff_freq:     parameter of lowpass filter, in Hz
    :param sampling_freq:   parameter of lowpass filter, in Hz
    :param order:           parameter of lowpass filter (default=5)
    
    """
    nyquist_freq = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, column)
    return filtered_data

def apply_savgol(column, window_length, poly_order):
    """apply Savitzky-Golay filter to one column of data
    
    Arguments:
    :param column:          one column of all data to be filtered, in dataframe
    :param window_length:   parameter of Savitzky-Golay filter
    :param poly_order:      parameter of Savitzky-Golay filter

    """
    # Check if the window length is appropriate for the column size
    if len(column) > window_length:
        return savgol_filter(column, window_length, poly_order)
    else:
        return column  # Return original if column too short for the window



def filter_and_smooth(args):
    """Filter and smooth the raw csv by lowpass filter and Savitzky-Golay filter, then
    saved the smoothed result to csv 

    Arguments:
    :param args: A tuple containing two variables: 
        - raw_csv_path:        location of the raw CSV of feat dataframe, including .csv extension
        - smoothed_csv_path:   location where the smoothed result is saved to, including .csv extension
    
    """
    raw_csv_path, smoothed_csv_path = args
    
    # if the target file exists, then skip it since it has been processed. Othrewise proceed
    if os.path.exists(smoothed_csv_path):
        logging.info(f"File {smoothed_csv_path} already processed, skipping.")
        return

    # load raw CSV to py-feat FEX
    input_prediction = read_feat(raw_csv_path)
    # logging.info(f"Running detection for file {video_path}")
    logging.info(f"Running smoothing for file {raw_csv_path}")

    df_smooth = input_prediction.copy()

    # apply lowpass filter
    cutoff_freq = 6  # Set the cutoff frequency (Hz)
    sampling_freq = 29.97 # Set the sampling frequency (Hz)

    df_smooth[columns_to_filter] = df_smooth[columns_to_filter].apply(
        lambda x: apply_lowpass_filter(x, cutoff_freq, sampling_freq)
        )

    # apply Savitzky-Golay filter
    window_length = 11  # Must be odd
    poly_order = 5  # Must be less than window length

    df_smooth[columns_to_filter] = df_smooth[columns_to_filter].apply(
        apply_savgol, args=(window_length, poly_order)
        )

    df_smooth.to_csv(smoothed_csv_path)
    logging.info(f"Output saved to {smoothed_csv_path}")


def main():
    tasks = [] # multiprocessing pool

    for i in range(start_actor_num, end_actor_num):
        # Skip Actor 18 for RAVDESS song
        if i == 18 and isSong:
            continue
        # Format the folder name with leading zeros
        folder_name = f"Actor_{i:02}" 
        # Set the raw CSV folder path and smoothed CSV output folder path
        raw_csv_folder_path = os.path.join(tracked_path, folder_name)
        smoothed_csv_folder_path = os.path.join(smoothed_path, folder_name)

        # create folder for smoothed CSV output if the folder does not exist
        if not os.path.exists(smoothed_csv_folder_path):
            os.makedirs(smoothed_csv_folder_path)
            logging.info(f"Created folder {smoothed_csv_folder_path}")
        
        # traverse all raw CSV files in the actor folder
        for file_name in os.listdir(raw_csv_folder_path):
            # complete the CSV path
            raw_csv_path = os.path.join(raw_csv_folder_path, file_name)
            # extract RAVDESS coding, without file extension and path
            video_basename = os.path.splitext(os.path.basename(raw_csv_path))[0]
            # naming smoothed CSV file to the according video
            smoothed_csv_path = os.path.join(smoothed_csv_folder_path, f"{video_basename}.csv")
            
            # if the smoothed CSV file does not exists, then filter and smooth the raw CSV
            # otherwise skip this file since it has been processed
            tasks.append((raw_csv_path, smoothed_csv_path))
    
    with Pool(processes=num_processes) as pool:
        pool.map(filter_and_smooth, tasks)

if __name__ == '__main__':
    main()