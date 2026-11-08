# %% --- Load Modules -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pandas import concat, DataFrame, read_csv
from pathlib import Path
from nilearn.image import load_img
from numpy import arange, round, searchsorted

from _MT_functions import get_tasks, get_runs, get_subjects

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = ArgumentParser(description="???")

ap.add_argument("--data", required=False, default=Path(__file__).parent.parent/"data", help="???")
ap.add_argument("--move_txt_files", required=False, default=False, help="???")

args = ap.parse_args()

pat_data = Path(__file__).parent.parent/"data"
boo_move_txt_files = args.move_txt_files

# %% --- Load Dara --------------------------------------------------------------------------------

lis_pat_subjects = get_subjects(pat_data=pat_data)
lis_runs = get_runs()
lis_tasks = get_tasks()

# TODO: do not hard code, make dynamic
dic_task_events = {"EMOTION":["neut", "fear"],
                   "GAMBLING":["loss_event", "neut_event", "win_event"],
                   "LANGUAGE":["story", "math"],
                   "MOTOR":["rh", "lf", "t", "rf", "lh"],
                   "RELATIONAL":["relation", "match"],
                   "SOCIAL":["mental", "rnd"],
                   "WM":["0bk_body", "0bk_faces", "0bk_tools", "0bk_places", "2bk_body", "2bk_faces", "2bk_tools", "2bk_places"]}

# %% ---

print("\nGenerating block csv-files...\n")

for str_task in lis_tasks:

    lis_events = dic_task_events[str_task]

    for pat_subject in lis_pat_subjects:

        str_subject = pat_subject.name
        int_subject_id = int(str_subject.removeprefix("sub-"))
        pat_subject_func = pat_subject/"func"
        pat_subject_func_events = pat_subject_func/"task_event_txt_files"

        for str_run in lis_runs:

            df_event_blocks = DataFrame(columns=["subject", "task", "run", "event_type", "onset", "duration", "end"])

            for str_event in lis_events:

                df_event = read_csv(pat_subject_func_events/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_{str_event}.txt", sep="\t", header=None)

                df_event = df_event.rename({0:"onset", 1:"duration"}, axis=1)
                df_event.insert(loc=0, value=int_subject_id, column="subject")
                df_event.insert(loc=1, value=str_task, column="task")
                df_event.insert(loc=2, value=str_run, column="run")
                df_event.insert(loc=3, value=str_event, column="event_type")
                df_event = df_event.drop(columns=2)

                df_event_blocks = concat([df_event_blocks, df_event], axis=0)

            df_event_blocks["end"] = df_event_blocks["onset"] + df_event_blocks["duration"]
            df_event_blocks = df_event_blocks.sort_values("onset").reset_index(drop=True)
            df_event_blocks.index += 1

            pat_block_file = pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_blocks.csv"

            if str_task == "WM":

                df_event_blocks["event_type"] = df_event_blocks["event_type"].str.split("_").str[1]

            df_event_blocks.to_csv(pat_block_file, index=False)

            print(f"File saved at: {pat_block_file}")

# %% --- Create the Scans CSV-File ------------------------------------------------------------------------------------

print("\nGenerating scan csv-files...\n")

for pat_subject in lis_pat_subjects:

    str_subject = pat_subject.name
    int_subject_id = int(str_subject.removeprefix("sub-"))
    pat_subject_func = pat_subject/"func"

    for str_run in lis_runs:

        for str_task in lis_tasks:

            print(f"\033[1mSubject: {str_subject}, Run: {str_run}, Task: {str_task}\033[0m")

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
                int_idx_end = searchsorted(a=arr_scan_times_rounded, v=flo_block_end, side="right") # TODO: + 1 like in paper? causes error in some subjects so -> if out of bounds, dont add + 1

                # store the actual value (time in s) of these scans
                if flo_block_onset >= arr_scan_times_rounded[-1]:

                    continue

                else:

                    flo_scan_onset = arr_scan_times_rounded[int_idx_onset]

                # use the last scan if the end of the block is later than the last scan (otherwise it would cause an index error)
                if flo_block_end > arr_scan_times_rounded[-1]: 

                    flo_scan_end = arr_scan_times_rounded[-1]

                else:

                    flo_scan_end = arr_scan_times_rounded[int_idx_end]

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

            # TODO: drop the first tr?
            # df_scans = df_scans.drop(labels=0, axis=0)
            
            pat_scan_file = pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_scans.csv"

            df_scans.to_csv(pat_scan_file, index=False)

            print(f"File saved at: {pat_scan_file}")

