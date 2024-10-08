﻿# FELT-dataset

## Download RAVDESS

1. Visit RAVDESS [Zenodo page](https://zenodo.org/records/1188976).
2. Under 'Files' section, you can download RAVDESS song videos (named from `Video_Song_Actor_01.zip` to `Video_Song_Actor_24.zip`) and/or speech videos (named from `Video_Speech_Actor_01.zip` to `Video_Speech_Actor_24.zip`).
3. Unzip all of them to a folder. In the following steps and scripts, I will use `F:\RAVDESS\` to store all the videos.
4. The folder structure after unzipping should looks like this:
> 
    F:\RAVDESS
    ├─ Actor_01
    ├─ Actor_02
    ├─ Actor_03
    ├─ Actor_04
    ├─ Actor_05
    ├─ Actor_06
    ├─ Actor_07
    ├─ Actor_08
    ├─ Actor_09
    ├─ Actor_10
    ├─ Actor_11
    ├─ Actor_12
    ├─ Actor_13
    ├─ Actor_14
    ├─ Actor_15
    ├─ Actor_16
    ├─ Actor_17
    ├─ Actor_18
    ├─ Actor_19
    ├─ Actor_20
    ├─ Actor_21
    ├─ Actor_22
    ├─ Actor_23
    └─ Actor_24
>

## Set up Py-Feat

1. Create a virtual envirionment. Py-feat currently support up to Python 3.9 `python39 -m venv F:\feat-venv`
2. Navigate to venv directory `cd F:\feat-venv`
3. Activate the venv `.\Scripts\Activate.ps1`
4. Install Py-feat `pip install py-feat`
5. Install [PyTorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-9) `pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121`
6. Modify Py-feat Detector.py: At `F:\feat-venv\Lib\site-packages\feat\detector.py`, modify `detect_identity()` function's return value from `return self._convert_detector_output(facebox, face_embeddings.numpy())` to `return self._convert_detector_output(facebox, face_embeddings.detach().numpy())` 

## Run scripts

To start, navigate to venv directory `cd F:\feat-venv` and clone this repo `git clone https://github.com/harveyliao/Py-feat-RAVDESS.git F:\feat-venv\Py-feat-RAVDESS`

### (Optional) remove video-only files

Video-only files contain same graphical information as Audio-video files, removing them can reduce processing time by half.

To remove video-only files, run `Py-feat-RAVDESS\src\utils\remove_video_only_files.py`

### Step 1: Run Py-feat detector

Run `Py-feat-RAVDESS\src\1_run_py-feat_detector.py`, then select configuration by numbers in console. Default configuration is to process all files in `F:\RAVDESS` folder. 

You can find the logging file at `run_detector.log`. The Py-feat detection result is saved as CSV files under `F:\raw_motion` 

### Step 2: Check NaN and interpolate

Run `Py-feat-RAVDESS\src\2_interpolate_motion.py`. 

This script checks for NA/NaN values and fills them by propagating the last valid observation to next valid. You can find the logging file at `chech_null_and_interpolate.log`. The interpolated data will overwrite those files in `F:\raw_motion` that has NA/NaN.   

### Step 3: Filter and smooth 

Run `Py-feat-RAVDESS\src\3_filter_and_smooth_data.py`

Files in `F:\raw_motion` is filtered and smoothed, then saved to `F:\smoothed_motion`. You can find the logging file at `filter_and_smooth_data.log`

### Step 4: Visualization

Install required library `pip install imageio[ffmpeg]`

#### Change landmark in Overlay from white to blue

Modify file `F:\feat-venv\Lib\site-packages\feat\data.py`: In function `plot_detections()`, find the line `color = "w"` and change it to `color = "b"`, save the file. Now the line face will be drawn in blue in Overlay.

#### draw Overlay videos

This script draws an overlay over RAVDESS videos. The overlay contains landmarks, head pose, facebox.

Run `Py-feat-RAVDESS\src\4_visualize_overlay.py`. It read from `F:\smoothed_motion` and save overlay videos to `F:\smoothed_video\Overlay\`. You can find the logging file at `draw_overlay.log`.

#### draw Landmark videos

This script produces videos containing landmarks and facebox, eliminating translational motion.

Run `Py-feat-RAVDESS\src\4_visualize_landmark.py`. It read from `F:\smoothed_motion` and save landmark videos to `F:\smoothed_video\Landmark\`. You can find the logging file at `draw_landmark.log`.

#### draw Action Units animation

This script draw Action Units (AU) animations. Note that Py-feat was unable to generate video frames for AU animation for about 10% of the frames for all video. Scripts drops those blank frames to have a more consistent result. Frames dropped are spread evenly across the timeline.

Run `Py-feat-RAVDESS\src\4_visualize_AU.py`. It read from `F:\smoothed_motion` and save AU animation to `F:\smoothed_video\ActionUnit\`. You can find the logging file at `draw_au.log`.


## Project folder structure


>
    F:\
    ├── RAVDESS                 # videos
    ├── feat-venv               # venv
    │   └── Py-feat-RAVDESS     # this repo
    ├── raw_motion              # output of step 1, then interpolated at step 2
    ├── smoothed_motion         # step 3 outputs
    └── smoothed_video          # step 4 outputs
        ├── ActionUnit          # action unit animation
        ├── Landmark            # landmark videos
        └── Overlay             # overlay videos
>

