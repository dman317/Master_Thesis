# %%

#TODO write text for 'help' and explain a bit, call the expected data location

import argparse
from pathlib import Path
import shutil

str_data_path = Path.cwd().parent/"data"

lis_files =["brainmask_LR_fs.2.nii.gz", "brainmask_RL_fs.2.nii.gz", "fear_LR.txt", "fear_RL.txt", \
            "neutral_LR.txt", "neutral_RL.txt", "tfMRI_EMOTION_LR.nii.gz","tfMRI_EMOTION_RL.nii.gz"]

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Should be the first script to be executed. Checks for completeness of the downloaded data and removes subjects with incomplete data (if --remove == TRUE).")
ap.add_argument("--data_path", required=False, default=str_data_path, help=f"path to the 'data' folder, default: ./../data")
ap.add_argument("--remove", required=False, default=False, help="remove directories with incomplete data?")

args = ap.parse_args()

str_data_path = str(args.data_path)
str_data_path = Path(str_data_path)

bool_remove = eval(str(args.remove))

# %% ---

print("")
print(f"Path: {str_data_path}")

lis_subject_paths = [folder for folder in str_data_path.iterdir() if folder.name.isdigit()]

print("")
print("--- Checking if folders contain required files ---")
print("")

lis_subjects_incomplete = []

for subject_path in lis_subject_paths:

    for file in lis_files:

        file_path = subject_path/file

        if file_path.exists():

            print(f"Subject: {subject_path.name}, File: {file:<30} \33[92mfile found\033[0m")

        else:

            print(f"Subject: {subject_path.name}, File: {file:<30} \033[91mFILE MISSING\033[0m")

            if subject_path.name not in lis_subjects_incomplete:
                
                lis_subjects_incomplete.append(subject_path.name)

print("")

int_n_subjects_incomplete = len(lis_subjects_incomplete)

# %%

print("--- Finished file check ---")
print("")
print(f"Subjects with missing files: {int_n_subjects_incomplete}")

if bool_remove and lis_subjects_incomplete:

    print("")
    print("----- Removing subject folders with missing data... -----")
    print("")

    for subject in lis_subjects_incomplete:

        subject_path = str_data_path/subject

        shutil.rmtree(subject_path)

        print (f"Subject: {subject} removed")

if int_n_subjects_incomplete > 0:

    print(f"Subjects removed: {bool_remove}")

print("")

