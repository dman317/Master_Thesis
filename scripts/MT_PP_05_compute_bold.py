# %% 

from argparse import ArgumentParser
from numpy import arange, float16, isin, savez_compressed, round, where
from pandas import read_csv
from nilearn.image import load_img, index_img
from nilearn.masking import apply_mask
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torch import tensor, save, float16 as to_float16, long

from _MT_functions import get_runs, get_subjects, get_tasks, get_codes

pat_data = Path.cwd().parent/"data_raw"

lis_subjects = get_subjects(pat_data = pat_data)
lis_runs = get_runs()
lis_tasks = get_tasks()
dic_codes = get_codes()

boo_force_new_files = True

img_mask_intersect = load_img(pat_data/"task-ALL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz")


# %% --- Handle Arguments -------------------------------------------------------------------------

#ap = ArgumentParser(description="insert description") #TODO

#ap.add_argument("--data_path", required=False, default=pat_data)

#args = ap.parse_args()

#pat_data_path = args.data_path

#pat_data = Path(pat_data_path)

# %% --- Handle Folders & Files -------------------------------------------------------------------

int_format_width = 87
str_format_file_found = "\033[92mfile found\033[0m"
str_format_file_generated = "\033[92mfile generated\033[0m"

for int_subject_i, pat_subject in enumerate(lis_subjects, start=1):

    str_subject = pat_subject.name
    pat_subject_func = pat_subject/"func"
    pat_subject_func_bold= pat_subject_func/"bold_files"

    print(f"Computing missing bold files for {str_subject} ({int_subject_i}/{len(lis_subjects)})...")

    if not pat_subject_func_bold.is_dir():
        
        pat_subject_func_bold.mkdir()

    print(f"Folder: {pat_subject_func_bold}")

    for str_run in lis_runs:

        for str_task in lis_tasks:

            # --- check .npz-file ----------------------------------------------------------------------

            str_file_npz = f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.npz"
            pat_file_npz = pat_subject_func_bold/str_file_npz

            str_file_pt = f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.pt"
            pat_file_pt = pat_subject_func_bold/str_file_pt

            if pat_file_npz.is_file() and pat_file_pt.is_file():

                boo_files_found = True

            else:

                boo_files_found = False


            # --- generate .npz and .pt-file if any is missing ------------------------------------

            if not boo_files_found or boo_force_new_files:

                img_task = load_img(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
                df_event_scans = read_csv(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_scans.csv")
                df_event_scans.insert(loc=len(df_event_scans.columns) - 1, column="event_type_code", value=df_event_scans["event_type"].map(dic_codes))

                int_n_scans = img_task.header.get_data_shape()[3]
                flo_t_r = img_task.header.get_zooms()[3]
                arr_scan_times_rounded = round(a=arange(int_n_scans) * flo_t_r, decimals=2)

                arr_bool_scan_matches = isin(element=arr_scan_times_rounded, test_elements=df_event_scans["tr"].values)
                img_task_filtered = index_img(imgs=img_task, index=arr_bool_scan_matches)

                # save data as .npz-file (for the SVM)
                arr_task_filtered_masked = apply_mask(imgs=img_task_filtered, mask_img=img_mask_intersect)
                scaler = StandardScaler()
                arr_task_filtered_masked_scaled = scaler.fit_transform(arr_task_filtered_masked)
                savez_compressed(file=pat_subject_func_bold/str_file_npz, data=arr_task_filtered_masked_scaled, labels=df_event_scans["event_type_code"].values)

                print(f"{str_file_npz:<{int_format_width}}{str_format_file_generated}")

                # save daza as .pt-file (for the CNN)
                ten_task_filtered = tensor(data=img_task_filtered.get_fdata(), dtype=to_float16)
                save(obj={"data":ten_task_filtered, "labels":df_event_scans["event_type"].values}, f=pat_subject_func_bold/str_file_pt)

                print(f"{str_file_pt:<{int_format_width}}{str_format_file_generated}")

            else:

                print(f"{str_file_npz:<{int_format_width}}{str_format_file_found}")
                print(f"{str_file_pt:<{int_format_width}}{str_format_file_found}")

    print("")

    #TODO take care of the mask issue
    #TODO come up with a better name for the saved npz-file
    #TODO make sure labels are correctly assigned
    #TODO come up with better prints and make them coherrent across SVM and DL scripts

# %%
