# %% 

from argparse import ArgumentParser
from numpy import arange, float16, savez_compressed, round, where
import pandas as pd
from nilearn.image import load_img, index_img
from nilearn.masking import apply_mask
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torch import tensor, save, float16 as to_float16, long

from scripts._MT_functions import get_subjects, get_mask_intersect

pat_data_path = Path.cwd().parent/"data"

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = ArgumentParser(description="insert description") #TODO

ap.add_argument("--data_path", required=False, default=pat_data_path)

args = ap.parse_args()

pat_data_path = args.data_path

pat_data_path = Path(pat_data_path)

# %% --- Load Essential Data ----------------------------------------------------------------------

lis_subjects = get_subjects(data_path=pat_data_path)
img_mask_intersect = get_mask_intersect(data_path=pat_data_path)

# %% --- Handle Folders & Files -------------------------------------------------------------------

int_format_width = 90
str_format_FILE_MISSING = "\033[91mFILE MISSING\033[0m"
str_format_file_found = "\033[92mfile found\033[0m"
str_format_file_generated = "\033[92mfile generated\033[0m"

print(f"File Path: {pat_data_path}/<subject>/bold")
print("")

for str_subject in lis_subjects:

    pat_subject_func_path = pat_data_path/str_subject/"func"
    pat_subject_bold_path = pat_data_path/str_subject/"func"/"bold"

    # Bold-Folder

    if not (pat_subject_bold_path).is_dir():

        pat_subject_bold_path.mkdir()

    # Bold-Files

    df_events = pd.read_csv("/home/daniel/Desktop/Python/MasterThesis/data/decoding.csv", sep=",").sort_values("tr")

    boo_missing_files_npz = False
    boo_missing_files_pt = False

    for run in ["LR", "RL"]:

        str_file_name_npz = f"{str_subject}_task-EMOTION_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.npz"
        str_file_name_pt = f"{str_subject}_task-EMOTION_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.pt"

        if not (pat_subject_bold_path/str_file_name_npz).is_file():

            boo_missing_files_npz = True

        else: 

            print(f"{str_file_name_npz:<{int_format_width}} {str_format_file_found}")

        if not (pat_subject_bold_path/str_file_name_pt).is_file():

            boo_missing_files_pt = True

        else:

            print(f"{str_file_name_npz:<{int_format_width}} {str_format_file_found}")

        if  boo_missing_files_npz or boo_missing_files_pt:
            
            img_func = load_img(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")

            int_n_scans = img_func.header.get_data_shape()[3]
            flo_t_r = img_func.header.get_zooms()[3]
            arr_scan_times = arange(int_n_scans) * flo_t_r

            arr_scan_times_rounded = round(a=arr_scan_times, decimals=2)
            arr_event_times_rounded = round(a=df_events["tr"].values, decimals=2)

            arr_match_indicies = []

            for event_time in arr_event_times_rounded:

                match = where(event_time == arr_scan_times_rounded)[0]
                
                arr_match_indicies.append(match[0])

            img_func_filtered = index_img(imgs=img_func, index=arr_match_indicies)
            arr_labels = df_events["true_state"].values
            
            # --- .npz ----------------------------------------------------------------------------

            if boo_missing_files_npz:

                print(f"{str_file_name_npz:<{int_format_width}} {str_format_FILE_MISSING}", end="\r")

                arr_func_filtered_masked = apply_mask(imgs=img_func_filtered, mask_img=img_mask_intersect).astype(float16)

                scaler = StandardScaler()
                arr_func_filtered_masked_scaled = scaler.fit_transform(arr_func_filtered_masked)

                pat_file_path_npz = pat_subject_bold_path/str_file_name_npz

                #savez_compressed(file=pat_file_path_npz, data_masked=arr_func_filtered_masked_scaled, labels=arr_labels)

                print(f"{str_file_name_npz:<{int_format_width}} {str_format_file_generated}")

            # --- .pt -----------------------------------------------------------------------------

            if boo_missing_files_pt:

                print(f"{str_file_name_pt:<{int_format_width}} {str_format_FILE_MISSING}", end="\r")

                #scaler = StandardScaler() #TODO
                #arr_func_filtered_scaled = scaler.fit_transform(img_func_filtered.get_fdata())

                ten_func_filtered_scaled = tensor(img_func_filtered.get_fdata(), dtype=to_float16)
                lon_labels = tensor(arr_labels, dtype=long)

                pat_file_path_pt = pat_subject_bold_path/str_file_name_pt

                #save(obj={"data":ten_func_filtered_scaled, "labels":lon_labels}, f=pat_file_path_pt)

                print(f"{str_file_name_pt:<{int_format_width}} {str_format_file_generated}")

    print("")

#TODO take care of the mask issue
#TODO come up with a better name for the saved npz-file
#TODO make sure labels are correctly assigned
#TODO come up with better prints and make them coherrent across SVM and DL scripts
# %%
