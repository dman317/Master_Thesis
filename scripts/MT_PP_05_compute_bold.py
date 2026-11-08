# %% 

from argparse import ArgumentParser
from numpy import arange, isin, savez, round
from pandas import read_csv
from nilearn.image import load_img, index_img
from nilearn.masking import apply_mask
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from _MT_functions import get_runs, get_subjects, get_tasks, get_codes

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = ArgumentParser(description="???")

ap.add_argument("--data", required=False, default=Path(__file__).parent.parent/"data", help="???")
ap.add_argument("--force_new_files", required=False, default=False, help="???")

args = ap.parse_args()

pat_data = args.data
boo_force_new_files = args.force_new_files

print("")
print("\33[1mArguments:\33[0m")
print(f"data: {pat_data}")
print(f"force_new_files: {boo_force_new_files}")
print("")

# %% --- Get Some Data ----------------------------------------------------------------------------

lis_subjects = get_subjects(pat_data = pat_data)
lis_runs = get_runs()
lis_tasks = get_tasks()
dic_codes = get_codes()

int_subjects_total = len(lis_subjects)

print(f"Subjects total: {int_subjects_total}")

str_mask_path = pat_data/"task-ALL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz"
img_mask = load_img(str_mask_path)

print(f"Mask loaded: {str_mask_path}")
print("")

# %% --- Handle Folders & Files -------------------------------------------------------------------

int_format_width = 87
str_format_file_found = "\033[92mfile found\033[0m"
str_format_file_generated = "\033[92mfile generated\033[0m"

for int_subject_i, pat_subject in enumerate(lis_subjects, start=1):

    str_subject = pat_subject.name
    pat_subject_func = pat_subject/"func"
    pat_subject_func_bold= pat_subject_func/"bold_files"

    print(f"\33[1m{str_subject} ({int_subject_i}/{len(lis_subjects)})\033[0m")

    if not pat_subject_func_bold.is_dir():
        
        pat_subject_func_bold.mkdir()

    print(f"\33[1m.npz-folder: {pat_subject_func_bold}\33[0m")

    for str_run in lis_runs:

        for str_task in lis_tasks:

            str_file_npz = f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.npz"
            pat_file_npz = pat_subject_func_bold/str_file_npz

            if not pat_file_npz.is_file() or boo_force_new_files:

                # load and transform data
                img_task = load_img(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
                df_event_scans = read_csv(pat_subject_func/f"{str_subject}_task-{str_task}_run-{str_run}_desc-EV_scans.csv")
                df_event_scans.insert(loc=len(df_event_scans.columns) - 1, column="event_type_code", value=df_event_scans["event_type"].map(dic_codes))

                # read out data and calculate time (in s) for each scan
                int_n_scans = img_task.header.get_data_shape()[3]
                flo_t_r = img_task.header.get_zooms()[3]
                arr_scan_times_rounded = round(a=arange(int_n_scans) * flo_t_r, decimals=2)

                # filter scans with irrelevant data out
                arr_bool_scan_matches = isin(element=arr_scan_times_rounded, test_elements=df_event_scans["tr"].values)
                img_task_filtered = index_img(imgs=img_task, index=arr_bool_scan_matches)

                # apply mask and transform data into the sample-feature format
                arr_task_filtered_masked = apply_mask(imgs=img_task_filtered, mask_img=img_mask)

                # standardize data
                scaler = StandardScaler()
                arr_task_filtered_masked_scaled = scaler.fit_transform(arr_task_filtered_masked)

                # save data as .npz-file
                savez(file=pat_file_npz, data=arr_task_filtered_masked_scaled, labels=df_event_scans["event_type_code"].values)

                print(f"{str_file_npz:<{int_format_width}}{str_format_file_generated}")

            else:

                print(f"{str_file_npz:<{int_format_width}}{str_format_file_found}")

    print("")


# %%
