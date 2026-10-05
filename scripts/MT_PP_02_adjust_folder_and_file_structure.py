
# %% ---

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
from _MT_functions import get_subjects
import regex

pat_data_path = Path.cwd().parent/"data_raw"

# %% ---

#ap = ArgumentParser()
#ap.add_argument("--path",required=False, default=pat_data_path, help="path to the folder that contains the str_subject subfolders")

#args = ap.parse_args()

#pat_data_path = str(args.path)
#pat_data_path = Path(pat_data_path)

# %% ---

list_pat_subjects = get_subjects(data_path=pat_data_path)
n_subjects = len(list_pat_subjects)

print("")
print(f"Path: {pat_data_path}")
print(f"Number of Subjects: {n_subjects}")
print("")

# %% --- Rename Subject Folders -------------------------------------------------------------------

for pat_subject in list_pat_subjects:

    if not pat_subject.name.startswith("sub-"):

        path_old = pat_data_path/pat_subject
        path_new = pat_data_path/f"sub-{pat_subject.name}"

        path_old.rename(target=path_new)

        list_pat_subjects = get_subjects(data_path=pat_data_path)

# %% --- Create 'anat'- and 'func'-Folders --------------------------------------------------------

str_name_anat_folder = "anat"
str_name_func_folder = "func"

for pat_subject in list_pat_subjects:

    if not (pat_subject/str_name_anat_folder).is_dir():

        (pat_subject/str_name_anat_folder).mkdir()
    
    if not (pat_subject/str_name_func_folder).is_dir():

        (pat_subject/str_name_func_folder).mkdir()

# %% --- Move Files to 'anat'- or 'func'-Folders --------------------------------------------------


for pat_subject in list_pat_subjects:

    lis_subject_files = [file for file in pat_subject.iterdir() if file.is_file()]

    for pat_file in lis_subject_files:

        if pat_file.name == "T1w.nii.gz":

            pat_subject_anat_file = pat_subject/str_name_anat_folder/pat_file.name

            pat_file.rename(target=pat_subject_anat_file)

        else:

            pat_subject_func_file = pat_subject/str_name_func_folder/pat_file.name

            pat_file.rename(pat_subject_func_file)

# %% --- Rename Files -----------------------------------------------------------------------------

lis_tasks = []
lis_runs = []

for pat_subject in list_pat_subjects:

    str_subject = pat_subject.name

    for pat_file in (pat_subject/str_name_func_folder).iterdir():

        str_file_name = pat_file.name

        lis_file_name_split = str_file_name.split("_")

        lis_file_name_split[1] = lis_file_name_split[1].removeprefix("task-")
        lis_file_name_split[2] = lis_file_name_split[2].removeprefix("run-")

        str_task = lis_file_name_split[1]
        str_run = lis_file_name_split[2]

        if str_task not in lis_tasks:

            lis_tasks.append(str_task)

        if str_run not in lis_runs:

            lis_runs.append(str_run)

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


# %% --- 

if "EMOTION" in lis_tasks:

    for pat_subject in list_pat_subjects:

        str_subject = pat_subject.name

        for run in lis_runs:

            data_neutral = pd.read_csv(pat_subject/"func"/f"{str_subject}_task-EMOTION_run-{run}_desc-EV_neut.txt", delimiter="\t", header=None)

            data_neutral.insert(loc=0, column="event_type", value="neut")

            data_fear = pd.read_csv(pat_subject/"func"/f"{str_subject}_task-EMOTION_run-{run}_desc-EV_fear.txt", delimiter="\t", header=None)
            
            data_fear.insert(loc=0, column="event_type", value="fear")
            
            data_summary = pd.concat([data_neutral, data_fear])

            data_summary.insert(loc=0, column="subject-id", value=str_subject.removeprefix("sub-"))
            data_summary.insert(loc=1, column="task", value="EMOTION")
            data_summary.insert(loc=2, column="run", value=run)
            data_summary = data_summary.rename(columns={0:"onset", 1:"duration", 2:"end"})
            data_summary["end"] = data_summary["onset"] + data_summary["duration"]
            data_summary = data_summary.sort_values("onset")
            data_summary["trial"] = range(len(data_summary))
            data_summary["trial"] += 1

            data_summary.to_csv(pat_subject/"func"/f"{str_subject}_task-EMOTION_run-{run}_desc-EV_summary.csv", index=False)

# %% --- Gambling ---------------------------------------------------------------------------------

if "GAMBLING" in lis_tasks:

    for pat_subject in list_pat_subjects:

        str_subject_name = pat_subject.name

        for str_run in lis_runs:

            df_gambling_loss_event = pd.read_csv(pat_subject/"func"/f"{str_subject_name}_task-GAMBLING_run-{str_run}_desc-EV_loss_event.txt", sep="\t", header=None)
            df_gambling_loss_event.insert(loc=0, column="Event", value="Loss")
            df_gambling_loss_event = df_gambling_loss_event.rename(columns={0:"onset", 1:"duration"})

            df_gambling_neut_event = pd.read_csv(pat_subject/"func"/f"{str_subject_name}_task-GAMBLING_run-{str_run}_desc-EV_neut_event.txt", sep="\t", header=None)
            df_gambling_neut_event.insert(loc=0, column="Event", value="Neut")
            df_gambling_neut_event = df_gambling_neut_event.rename(columns={0:"onset", 1:"duration"})

            df_gambling_win_event  = pd.read_csv(pat_subject/"func"/f"{str_subject_name}_task-GAMBLING_run-{str_run}_desc-EV_win_event.txt", sep="\t", header=None)
            df_gambling_win_event.insert(loc=0, column="Event", value="Win")
            df_gambling_win_event = df_gambling_win_event.rename(columns={0:"onset", 1:"duration"})

            df_gambling_events = pd.concat([df_gambling_loss_event, df_gambling_neut_event, df_gambling_win_event], axis=0).reset_index(drop=True)
            df_gambling_events = df_gambling_events.sort_values("onset")

            df_gambling_loss = pd.read_csv(pat_subject/"func"/f"{str_subject_name}_task-GAMBLING_run-{str_run}_desc-EV_loss.txt", sep="\t", header=None)
            df_gambling_loss = df_gambling_loss.rename(columns={0:"onset", 1:"duration"})

            df_gambling_win  = pd.read_csv(pat_subject/"func"/f"{str_subject_name}_task-GAMBLING_run-{str_run}_desc-EV_win.txt", sep="\t", header=None)
            df_gambling_loss = df_gambling_loss.rename(columns={0:"onset", 1:"duration"})

            np.where(df_gambling_loss["onset"] == df_gambling_loss_event["onset"]) #FIXME

# %%

if "MOTOR" in lis_tasks:

    lis_column_names = ["event", "onset", "duration", "no idea what this is, find out!"]

    for pat_subject in list_pat_subjects:

        str_subject_name = pat_subject.name

        lis_subject_motor_LR_event_files = []
        lis_subject_motor_RL_event_files = []

        df_motor_LR_events = pd.DataFrame(columns=lis_column_names)
        lis_subject_motor_RL_event_dfs = []

        for pat_file in (pat_subject/"func").iterdir():

            str_file = pat_file.name

            if regex.findall("MOTOR_run-LR", str_file) and str_file.endswith(".txt"):

                lis_subject_motor_LR_event_files.append(str_file)

            if regex.findall("MOTOR_run-RL", str_file) and str_file.endswith(".txt"):

                lis_subject_motor_RL_event_files.append(str_file)

        for str_file in lis_subject_motor_LR_event_files:

            str_event_name = regex.findall(pattern="desc-EV_(\\w+).txt", string=str_file)[0]

            df_motor_events = pd.read_csv(pat_subject/"func"/str_file, sep="\t", header=None)
            df_motor_events = df_motor_events.rename(columns={0:lis_column_names[1], 1:lis_column_names[2], 2:lis_column_names[3]})
            df_motor_events.insert(loc=0, column="event", value=str_event_name)

            df_motor_LR_events = pd.concat(objs=[df_motor_LR_events, df_motor_events], axis=0)

        df_motor_LR_events.to_csv(pat_subject/"func"/f"{str_subject_name}_task-MOTOR_run-LR_desc-EV_full")

        for str_file in lis_subject_motor_RL_event_files:

            str_event_name = regex.findall(pattern="desc-EV_(\\w+).txt", string=str_file)[0]

            df_motor_events = pd.read_csv(pat_subject/"func"/str_file, sep="\t", header=None)
            df_motor_events = df_motor_events.rename(columns={0:lis_column_names[1], 1:lis_column_names[2], 2:lis_column_names[3]})
            df_motor_events.insert(loc=0, column="event", value=str_event_name)

            df_motor_RL_events = pd.concat(objs=[df_motor_RL_events, df_motor_events], axis=0)

        df_motor_RL_events.to_csv(pat_subject/"func"/f"{str_subject_name}_task-MOTOR_run-RL_desc-EV_full")