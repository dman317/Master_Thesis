
# %% --- Load Modules -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from regex import search

from _MT_functions import get_subjects

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = ArgumentParser()
ap.add_argument("--data", required=False, default=Path(__file__).parent.parent/"data", help="???") # TODO. help

args = ap.parse_args()

pat_data = args.data

# %% --- Load Data --------------------------------------------------------------------------------

lis_pat_subjects = get_subjects(pat_data=pat_data)
int_n_subjects = len(lis_pat_subjects)

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

    str_subject = pat_subject.name

    lis_subject_files = [file for file in pat_subject.iterdir() if file.is_file()]

    for pat_file in lis_subject_files:

        if pat_file.name == "T1w.nii.gz":

            pat_subject_anat_file = pat_subject/str_name_anat_folder/f"{str_subject}_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz"

            pat_file.rename(target=pat_subject_anat_file)

        else:

            pat_subject_func_file = pat_subject/str_name_func_folder/pat_file.name

            pat_file.rename(pat_subject_func_file)

# %% --- Rename Func Files -----------------------------------------------------------------------------

for pat_subject in lis_pat_subjects:

    str_subject = pat_subject.name
    pat_subject_func = pat_subject/str_name_func_folder

    for pat_file in pat_subject_func.iterdir():

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

        if str_file_name.endswith(".txt"):

            str_file_ending = search(pattern=r"(?:RL|LR)_(.+).txt", string=str_file_name)[1]

            if str_file_ending.startswith("desc-EV_"):

                str_file_ending = str_file_ending.removeprefix("desc-EV_")

            pat_new_path = pat_file.parent/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_{str_file_ending}.txt"

            if not pat_new_path.is_file():

                pat_file.rename(pat_new_path)

# %% --- Move txt Files ---------------------------------------------------------------------------

for pat_subject in lis_pat_subjects:

    pat_subject_func = pat_subject/"func"
    pat_subject_func_events = pat_subject_func/"task_event_txt_files"

    if not pat_subject_func_events.is_dir():

        pat_subject_func_events.mkdir()

    for pat_file in pat_subject_func.iterdir():

        str_file = pat_file.name

        if str_file.endswith(".txt"):

            pat_file.rename(target=pat_subject_func_events/str_file)
            
# %% --- Move Mask Files --------------------------------------------------------------------------

for pat_subject in lis_pat_subjects:

    pat_subject_func = pat_subject/"func"
    pat_subject_func_masks = pat_subject_func/"task_mask_files"

    if not pat_subject_func_masks.is_dir():

        pat_subject_func_masks.mkdir()

    for pat_file in pat_subject_func.iterdir():

        str_file = pat_file.name

        if str_file.endswith("brain_mask.nii.gz"):

            pat_file.rename(target=pat_subject_func_masks/str_file)
