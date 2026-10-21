
# %% ---

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
from _MT_functions import get_subjects

pat_data = Path("/home/daniel/Desktop/Python/MasterThesis/data")

# %% ---

#ap = ArgumentParser()
#ap.add_argument("--path",required=False, default=pat_data_path, help="path to the folder that contains the str_subject subfolders")

#args = ap.parse_args()

#pat_data_path = str(args.path)
#pat_data_path = Path(pat_data_path)

# %% ---

lis_pat_subjects = get_subjects(pat_data=pat_data)
n_subjects = len(lis_pat_subjects)

print("")
print(f"Path: {pat_data}")
print(f"Number of Subjects: {n_subjects}")
print("")

# %% --- Rename Subject Folders -------------------------------------------------------------------

for pat_subject in lis_pat_subjects:

    str_subject = pat_subject.name

    if not str_subject.startswith("sub-"):

        path_old = pat_data/str_subject
        path_new = pat_data/f"sub-{str_subject}"

        path_old.rename(target=path_new)

        lis_pat_subjects = get_subjects(pat_data=pat_data)

# %% --- Create 'anat'- and 'func'-Folders --------------------------------------------------------

str_name_anat_folder = "anat"
str_name_func_folder = "func"

for pat_subject in lis_pat_subjects:

    if not (pat_subject/str_name_anat_folder).is_dir():

        (pat_subject/str_name_anat_folder).mkdir()
    
    if not (pat_subject/str_name_func_folder).is_dir():

        (pat_subject/str_name_func_folder).mkdir()

# %% --- Move Files to 'anat'- or 'func'-Folders --------------------------------------------------


for pat_subject in lis_pat_subjects:

    lis_subject_files = [file for file in pat_subject.iterdir() if file.is_file()]

    for pat_file in lis_subject_files:

        if pat_file.name == "T1w.nii.gz":

            pat_subject_anat_file = pat_subject/str_name_anat_folder/pat_file.name

            pat_file.rename(target=pat_subject_anat_file)

        else:

            pat_subject_func_file = pat_subject/str_name_func_folder/pat_file.name

            pat_file.rename(pat_subject_func_file)

# %% --- Rename Files -----------------------------------------------------------------------------

for pat_subject in lis_pat_subjects:

    str_subject = pat_subject.name

    for pat_file in (pat_subject/str_name_func_folder).iterdir():

        str_file_name = pat_file.name  # get file name

        lis_file_name_split = str_file_name.split("_")  # split file name into segments

        str_task = lis_file_name_split[1].removeprefix("task-")
        str_run = lis_file_name_split[2].removeprefix("run-")

        if str_file_name.endswith("tfMRI.nii.gz"):

            pat_new_path = pat_file.parent/f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"

            if not pat_new_path.is_file():

                pat_file.rename(pat_new_path)

        if str_file_name.endswith ("brainmask_fs.2.nii.gz"):

            pat_new_path = pat_file.parent/f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz"

            if not pat_new_path.is_file():

                pat_file.rename(pat_new_path)

        if str_file_name.endswith(".txt"):  #FIXME

            str_file_ending = str_file_name.split(str_run)[1].removeprefix("_")

            pat_new_path = pat_file.parent/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_{str_file_ending}"

            if not pat_new_path.is_file():

                pat_file.rename(pat_new_path)
