import pandas as df
import os
from feat.utils.io import read_feat

# Configuration
csv_path = "F:/tracked_song/"
start_actor_num = 1
end_actor_num = 25
isSong = True

def has_null_values(df):
  """Checks if a DataFrame contains any null values.

  Args:
    df: The pandas DataFrame to check.

  Returns:
    True if there are null values, False otherwise.
  """
  return df.isnull().values.any()

def main():
    csv_contain_null = []
    for i in range(start_actor_num, end_actor_num):
        if i == 18 and isSong:
            continue
        folder_name = f"Actor_{i:02}" 
        smoothed_csv_folder_path = os.path.join(csv_path, folder_name)
        
        for file_name in os.listdir(smoothed_csv_folder_path):
            smoothed_csv_path = os.path.join(smoothed_csv_folder_path, file_name)
            video_basename = os.path.splitext(os.path.basename(smoothed_csv_path))[0]
            
            fex_dataframe = read_feat(smoothed_csv_path)
            if has_null_values(fex_dataframe):
                csv_contain_null.append(video_basename)
    
    print(f"in {csv_path}, there are {len(csv_contain_null)} files contains null: ", end='')
    print(csv_contain_null)

if __name__ == '__main__':
    main()
