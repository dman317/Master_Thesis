# %%

from pandas import concat, DataFrame, read_csv, Series
from pathlib import Path
from nilearn.image import load_img
from numpy import arange, round, searchsorted

from _MT_functions import get_tasks, get_runs, get_subjects

#pat_data = Path.cwd().parent/"data_raw"
pat_data = Path("/home/daniel/Desktop/Python/MasterThesis/data")
lis_pat_subjects = get_subjects(pat_data=pat_data)
lis_runs = get_runs()
lis_tasks = get_tasks()
boo_csv_each_event = False
boo_move_txt_files = True

dic_task_events = {"EMOTION":["neut", "fear"],
                   "GAMBLING":["loss_event", "neut_event", "win_event"], #TODO change to "loss", "neut", "win"
                   "LANGUAGE":["story", "math"],
                   "MOTOR":["rh", "lf", "t", "rf", "lh"],
                   "RELATIONAL":["relation", "match"],
                   "SOCIAL":["mental", "rnd"],  #TODO change to "mental", "other"
                   "WM":["0bk_body", "0bk_faces", "0bk_tools", "0bk_places", "2bk_body", "2bk_faces", "2bk_tools", "2bk_places"]}

# %% --- Create the Summary CSV-File --------------------------------------------------------------

for str_task in lis_tasks:

    lis_events = dic_task_events[str_task]

    for pat_subject in lis_pat_subjects:

        str_subject = pat_subject.name
        int_subject_id = int(str_subject.removeprefix("sub-"))
        pat_subject_func = pat_subject/"func"
        pat_subject_func_events = pat_subject_func/"event_txt_files"

        for str_run in lis_runs:

            df_event_summary = DataFrame(columns=["subject", "task", "run", "event_type", "onset", "duration", "end", "trial"])

            for str_event in lis_events:

                try:

                    df_event = read_csv(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_{str_event}.txt", sep="\t", header=None)
                except:

                    df_event = read_csv(pat_subject_func_events/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_{str_event}.txt", sep="\t", header=None)
                
                df_event = df_event.rename({0:"onset", 1:"duration"}, axis=1)
                df_event.insert(loc=0, value=int_subject_id, column="subject")
                df_event.insert(loc=1, value=str_task, column="task")
                df_event.insert(loc=2, value=str_run, column="run")
                df_event.insert(loc=3, value=str_event, column="event_type")
                df_event = df_event.drop(columns=2)

                if boo_csv_each_event:

                    df_event.to_csv(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_{str_event}_blocks.csv", index=False)

                df_event_summary = concat([df_event_summary, df_event], axis=0)

            df_event_summary["end"] = df_event_summary["onset"] + df_event_summary["duration"]
            df_event_summary = df_event_summary.sort_values("onset").reset_index(drop=True)
            df_event_summary["trial"] = range(len(df_event_summary))
            df_event_summary.index += 1

            df_event_summary.to_csv(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_blocks.csv", index=False)

# %% --- Create the Scans CSV-File ------------------------------------------------------------------------------------

for pat_subject in lis_pat_subjects:

    str_subject = pat_subject.name
    int_subject_id = int(str_subject.removeprefix("sub-"))
    pat_subject_func = pat_subject/"func"

    for str_run in lis_runs:

        for str_task in lis_tasks:

            df_scans = DataFrame(columns=["subject", "task", "run", "event_type", "tr"])

            img_task = load_img(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
            df_events = read_csv(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_blocks.csv")

            int_n_scans = img_task.header.get_data_shape()[3]
            flo_t_r = img_task.header.get_zooms()[3]
            arr_scan_times = arange(int_n_scans) * flo_t_r

            arr_scan_times_rounded = round(a=arr_scan_times, decimals=2)

            for i_row, row in df_events.iterrows():

                flo_block_onset = round(a=df_events.loc[i_row, "onset"], decimals=2)
                flo_block_end = round(a=df_events.loc[i_row, "end"], decimals=2)
                str_block_event = row["event_type"]

                # find the index of the next scan after begin/end of the block
                int_idx_onset = searchsorted(a=arr_scan_times_rounded, v=flo_block_onset, side="right")
                int_idx_end = searchsorted(a=arr_scan_times_rounded, v=flo_block_end, side="right") + 1 # add 1 to include the next 2 scans after the block ends

                # store the actual value (time in s) of these scans
                flo_scan_onset = arr_scan_times_rounded[int_idx_onset]

                # use the last scan if the end of the block is later than the last scan (otherwise it would cause an index error)
                if not flo_block_end > arr_scan_times_rounded[-1]: 

                    flo_scan_end = arr_scan_times_rounded[int_idx_end]

                else:

                    flo_scan_end = arr_scan_times_rounded[-1]

                # get all the scans between the onset scan and the end scan
                arr_block_scan_times = arr_scan_times_rounded[(arr_scan_times_rounded >= flo_scan_onset) & (arr_scan_times_rounded <= flo_scan_end)]

                # create a dataframe, storing all the scans and their corresponding label
                df_block_scan_times = DataFrame({"event_type":str_block_event, "tr":arr_block_scan_times})

                df_block_scan_times = df_block_scan_times.drop(labels=0, axis=0) # ???

                # add the set of scan times and their labels to df_scans
                df_scans = concat(objs=[df_scans, df_block_scan_times])

            df_scans["subject"] = int_subject_id
            df_scans["task"] = str_task
            df_scans["run"] = str_run

            df_scans = df_scans.reset_index(drop=True)

            # drop the first tr (no idea why but they did it in the paper)
            # df_scans = df_scans.drop(labels=0, axis=0)
            
            df_scans.to_csv(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_scans.csv", index=False)

# %% ----------------------------------------------------------------------------------------------

if boo_move_txt_files:

    for pat_subject in lis_pat_subjects:

        pat_subject_func = pat_subject/"func"
        pat_subject_func_events = pat_subject_func/"event_txt_files"

        if not pat_subject_func_events.is_dir():

            pat_subject_func_events.mkdir()

        for pat_file in pat_subject_func.iterdir():

            str_file = pat_file.name

            if str_file.endswith(".txt"):

                pat_file.rename(target=pat_subject_func_events/str_file)

# %% ----------------------------------------------------------------------------------------------







df_emotion = read_csv("/home/daniel/Desktop/Python/MasterThesis/evaluating-deeplight-transfer/results/predictions/DeepLight/3D/pretrained-True/sub-100307/task-EMOTION_sub-100307_run-LR_desc-predictions_3D-DeepLight_pretrained-True.csv")
df_emotion.sort_values(by="tr")