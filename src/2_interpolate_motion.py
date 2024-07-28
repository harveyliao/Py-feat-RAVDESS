import pandas as pd
import os
import logging
from feat.utils.io import read_feat
from multiprocessing import Pool, cpu_count

# Configuration
csv_path = "F:/raw_motion/"
start_actor_num = 1
end_actor_num = 25

# Setup logging
logging.basicConfig(filename="chech_null_csv.log", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class FillNaNError(Exception):
    """Custom exception raised when ffill() fails to remove all null values in a DataFrame."""
    pass

def has_null_values(df: pd.DataFrame) -> bool:
    """Checks if a DataFrame contains any null values.

    Args:
    :param df: The pandas DataFrame to check.

    Returns:
        True if there are null values, False otherwise.
    """
    return df.isnull().values.any()

def process_actor(actor_num: int) -> list:
    """Process CSV files for a single actor folder.

    Iterate through all files under certain Actor's folder. call has_null_values() to each 
    file. if that file contains null, try to ffill() it. If success, write the filled version 
    back to original path, othrewise raise an FillNaNError.

    Args:
    :param actor_num: The actor number to process.

    Returns:
    :return csv_contain_null: A list of CSV file basenames containing null values.
    """
    csv_contain_null = []
    folder_name = f"Actor_{actor_num:02}"
    smoothed_csv_folder_path = os.path.join(csv_path, folder_name)

    for file_name in os.listdir(smoothed_csv_folder_path):
        smoothed_csv_path = os.path.join(smoothed_csv_folder_path, file_name)
        video_basename = os.path.splitext(os.path.basename(smoothed_csv_path))[0]

        logging.info(f"Now loading file {smoothed_csv_path}")
        fex_dataframe = read_feat(smoothed_csv_path)

        if has_null_values(fex_dataframe):
            csv_contain_null.append(video_basename)
            logging.info(f"Null found at file {smoothed_csv_path}, call ffill()")
            fex_dataframe = fex_dataframe.ffill()
            if not has_null_values(fex_dataframe):
                fex_dataframe.to_csv(smoothed_csv_path)
                logging.info(f"ffill() completed, now saved to {smoothed_csv_path}")
            else:
                logging.error(f"Failed to ffill() file {smoothed_csv_path}")
                raise FillNaNError(f"Failed to ffill() file {smoothed_csv_path}")

    return csv_contain_null

def main():
    actor_nums = range(start_actor_num, end_actor_num)
    with Pool(cpu_count()) as pool:
        results = pool.map(process_actor, actor_nums)

    # Flatten the list of lists
    csv_contain_null = [item for sublist in results for item in sublist]

    print(f"There were {len(csv_contain_null)} files containing null", end='')
    if csv_contain_null:
        print(": ", end='')
        print(csv_contain_null)
        print("ffill() have been successfully applied to those files.")

if __name__ == '__main__':
    main()
