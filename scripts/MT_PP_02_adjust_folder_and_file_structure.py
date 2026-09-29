
# %% ---

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

pat_data_path = Path.cwd().parent/"data"

# %% ---

ap = ArgumentParser()
ap.add_argument("--path",required=False, default=pat_data_path, help="path to the folder that contains the str_subject subfolders")

args = ap.parse_args()

pat_data_path = str(args.path)
pat_data_path = Path(pat_data_path)

# %% ---

print("Path found!")

list_subjects = [str_subject.name for str_subject in list(pat_data_path.iterdir()) if str_subject.name.isdigit()]
n_subjects = len(list_subjects)

print("")
print(f"Path: {pat_data_path}")
print(f"Number of Subjects: {n_subjects}")
print("")

# %% --- Rename str_subject Folders -------------------------------------------------------------------

for str_subject in list_subjects:

    if not str_subject.startswith("sub-"):

        path_old = pat_data_path/str_subject
        path_new = pat_data_path/f"sub-{str_subject}"

        path_old.rename(target=path_new)

# %% --- Create 'anat'- and 'func'-Folders -----------------------------------------------------------

list_subjects = [str_subject.name for str_subject in list(pat_data_path.iterdir()) if str_subject.name.startswith("sub-")]

str_name_anat_folder = "anat"
str_name_func_folder = "func"

for str_subject in list_subjects:

    pat_path_subject = pat_data_path/str_subject

    if not (pat_path_subject/str_name_anat_folder).is_dir():

        (pat_path_subject/str_name_anat_folder).mkdir()
    
    if not (pat_path_subject/str_name_func_folder).is_dir():

        (pat_path_subject/str_name_func_folder).mkdir()

# %% --- Rename and Move Files --------------------------------------------------------------------

for str_subject in list_subjects:

    pat_subject_path = pat_data_path/str_subject
    pat_subject_anat_path = pat_subject_path/str_name_anat_folder
    pat_subject_func_path = pat_subject_path/str_name_func_folder

    lis_subject_files = [file.name for file in pat_subject_path.iterdir() if file.is_file()]

    for file in lis_subject_files:

        if file == 'T1w.nii.gz':

            (pat_subject_path/file).rename(pat_subject_anat_path/f"{str_subject}_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz")
        
        if file == 'brainmask_LR_fs.2.nii.gz':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz")

        if file == 'brainmask_RL_fs.2.nii.gz':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz")

        if file ==  'tfMRI_EMOTION_LR.nii.gz':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")

        if file ==  'tfMRI_EMOTION_RL.nii.gz':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")

        if file ==  'neutral_LR.txt':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-LR_descv-EV_neut.txt")

        if file ==  'neutral_RL.txt':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-RL_descv-EV_neut.txt")

        if file ==  'fear_LR.txt':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-LR_descv-EV_fear.txt")

        if file ==  'fear_RL.txt':

            (pat_subject_path/file).rename(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-RL_descv-EV_fear.txt")

# %% --- 

for str_subject in list_subjects:

    pat_subject_path = pat_data_path/str_subject
    pat_subject_func_path = pat_subject_path/str_name_func_folder

    for run in ["RL", "LR"]:

        data_neutral = pd.read_csv((pat_subject_func_path/f"{str_subject}_task-EMOTION_run-{run}_descv-EV_neut.txt"), delimiter="\t", header=None)
        
        data_neutral.insert(loc=0, column="event_type", value="neut")

        data_fear = pd.read_csv(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-{run}_descv-EV_fear.txt", delimiter="\t", header=None)
        
        data_fear.insert(loc=0, column="event_type", value="fear")
        
        data_summary = pd.concat([data_neutral, data_fear])

        data_summary.insert(loc=0, column="str_subject", value=str_subject[4:])
        data_summary.insert(loc=1, column="task", value="EMOTION")
        data_summary.insert(loc=2, column="run", value=run)
        data_summary = data_summary.rename(columns={0:"onset", 1:"duration", 2:"end"})
        data_summary["end"] = data_summary["onset"] + data_summary["duration"]
        data_summary = data_summary.sort_values("onset")
        data_summary["trial"] = range(len(data_summary))

        data_summary.to_csv(pat_subject_func_path/f"{str_subject}_task-EMOTION_run-{run}_desc-EV_summary.csv", index=False)

# %%
