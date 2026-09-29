
# %% --- Load Modules -----------------------------------------------------------------------------

from argparse import ArgumentParser
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import load_img
from nilearn.masking import apply_mask
from numpy import arange, savez_compressed, unique, reshape
from os import listdir, mkdir
from os.path import isdir, isfile, join
from pandas import Categorical, concat, DataFrame, read_csv
from pathlib import Path
from torch import float32 as to_float32, long, save, tensor

from scripts._MT_functions import get_subjects, get_mask_intersect

str_default_data_path = str(Path.cwd().parent/"data") #TODOD

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = ArgumentParser(description="insert description") #TODO

ap.add_argument("--data_path", required=False, default=str_default_data_path)

args = ap.parse_args()

str_data_path = str(args.data_path)
str_data_path = Path(str_data_path)

# %% --- Load Essential Data ----------------------------------------------------------------------

lis_subjects = get_subjects(data_path=str_data_path)
img_mask = get_mask_intersect(data_path=str_data_path)

# %% --- Checking Beta Directories & Files --------------------------------------------------------

print("Checking for Beta Directories and Files...")
print("")

lis_subjects_missing_npz = []
lis_subjects_missing_pt = []

for int_subject_i, str_subject in enumerate(lis_subjects, start=1):

    print(f"{str_subject} ({int_subject_i}/{len(lis_subjects)})")
    print("    checking for beta directory...", end="\r")
        
    str_path_subject_betas = join(str_data_path, str_subject, "func", "betas")

    if not isdir(str_path_subject_betas):
        
        print("    checking for beta directory...no directory found")

        mkdir(str_path_subject_betas)

        print(f"    New directory created at: {str_path_subject_betas}")

        lis_subjects_missing_npz.append(str_subject)
        lis_subjects_missing_pt.append(str_subject)

        print(f"    added '{str_subject}' to list of subjects with missing or incomplete betas")

    else:

        print("    checking for beta directory...directory found")
        print("    checking for beta files...", end="\r")

        int_file_count_npz = 0
        int_file_count_pt = 0

        for file in listdir(str_path_subject_betas):

            if file.endswith(".npz"):

                int_file_count_npz += 1

            if file.endswith(".pt"):

                int_file_count_pt += 1
        
        if not int_file_count_npz == 12:

            lis_subjects_missing_npz.append(str_subject)

        if not int_file_count_pt == 12:

            lis_subjects_missing_pt.append(str_subject)

        if (str_subject in lis_subjects_missing_pt) or (str_subject in lis_subjects_missing_npz):
                
            print("    checking for beta files...files missing or incomplete")
            print(f"    added '{str_subject}' to list of subjects with missing or incomplete betas")

        else:

            print("    checking for beta files...files found")

# --- Computation of Beta Files -------------------------------------------------------------------

if bool(lis_subjects_missing_npz) or bool(lis_subjects_missing_pt):

    print("")
    str_temp = "--- Computing Beta Files "
    print(str_temp + "-"*(100-len(str_temp)))
    print("")

    arr_subjects_with_incomplete_data = unique(lis_subjects_missing_npz + lis_subjects_missing_pt)
    int_n_subjects_with_incomplete_data = len(arr_subjects_with_incomplete_data)

    for int_i_subject, str_subject in enumerate(arr_subjects_with_incomplete_data, start=1):

        int_subject_id = str_subject.split("-")[1]
        str_path_subject_betas = join(str_data_path, str_subject, "func", "betas")

        print(f"{str_subject} ({int_i_subject}/{int_n_subjects_with_incomplete_data})")

        str_subject_func_path = join(str_data_path, str_subject, "func/")

        img_func_LR = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"))
        img_mask_LR = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz"))
        df_events_LR = read_csv(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-LR_desc-EV_summary.csv"), sep=",")

        img_func_RL = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"))
        img_mask_RL = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz"))
        df_events_RL = read_csv(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-RL_desc-EV_summary.csv"), sep=",")

        # --- Transform Data ----------------------------------------------------------------------

        int_n_scans_LR = img_func_LR.header.get_data_shape()[3]
        flo_t_r_LR = float(img_func_LR.header.get_zooms()[3])
        lis_frame_times_LR = arange(int_n_scans_LR) * flo_t_r_LR

        int_n_scans_RL = img_func_RL.header.get_data_shape()[3]
        flo_t_r_RL = float(img_func_RL.header.get_zooms()[3])
        lis_frame_times_RL = arange(int_n_scans_RL) * flo_t_r_RL

        # --- Setup Unique Block Conditions -------------------------------------------------------

        df_labels_LR = DataFrame()
        df_labels_LR["onset"] = df_events_LR["onset"]
        df_labels_LR["duration"] = df_events_LR["duration"]
        df_labels_LR["trial_type"] = df_events_LR["event_type"]

        int_counter_neut = 1
        int_counter_fear = 1

        for row in df_labels_LR.iterrows():

            row_i = row[0]
            row = row[1]

            if row["trial_type"] == "neut":

                df_labels_LR.loc[row_i, "trial_type"] = f"neut_{int_counter_neut}"

                int_counter_neut += 1
            
            if row["trial_type"] == "fear":

                df_labels_LR.loc[row_i, "trial_type"] = f"fear_{int_counter_fear}"

                int_counter_fear += 1

        df_labels_RL = DataFrame()
        df_labels_RL["onset"] = df_events_RL["onset"]
        df_labels_RL["duration"] = df_events_RL["duration"]
        df_labels_RL["trial_type"] = df_events_RL["event_type"]

        int_counter_neut = 4
        int_counter_fear = 4

        for row_i, row in df_labels_RL.iterrows():

            if row["trial_type"] == "neut":

                df_labels_RL.loc[row_i, "trial_type"] = f"neut_{int_counter_neut}"

                int_counter_neut += 1
            
            if row["trial_type"] == "fear":

                df_labels_RL.loc[row_i, "trial_type"] = f"fear_{int_counter_fear}"

                int_counter_fear += 1

        # --- GLM ---------------------------------------------------------------------------------

        dm_LR = make_first_level_design_matrix(frame_times=lis_frame_times_LR, events=df_labels_LR)
        dm_RL = make_first_level_design_matrix(frame_times=lis_frame_times_RL, events=df_labels_RL)

        str_flm_hrf_model = "spm"

        glm_fl_LR = FirstLevelModel(t_r=flo_t_r_LR,
                                    hrf_model=str_flm_hrf_model,
                                    mask_img=img_mask_LR)
        
        glm_fl_LR.fit(run_imgs=img_func_LR, events=df_labels_LR)

        glm_fl_RL = FirstLevelModel(t_r=flo_t_r_RL,
                                    hrf_model=str_flm_hrf_model,
                                    mask_img=img_mask_RL)
        
        glm_fl_RL.fit(run_imgs=img_func_RL, events=df_labels_RL)

        lis_betas_LR = []
        lis_labels_LR = []

        for condition in df_labels_LR["trial_type"]:

            img_betas_LR = glm_fl_LR.compute_contrast(contrast_def=condition, output_type="effect_size")

            lis_betas_LR.append(img_betas_LR)
            lis_labels_LR.append(condition.split("_")[0])

        lis_betas_RL = []
        lis_labels_RL = []

        for condition in df_labels_RL["trial_type"]:

            img_betas_RL = glm_fl_RL.compute_contrast(contrast_def=condition, output_type="effect_size")

            lis_betas_RL.append(img_betas_RL)
            lis_labels_RL.append(condition.split("_")[0])
        
        lis_subject_betas = lis_betas_LR + lis_betas_RL

        df_labels_LR["run"] = "LR"
        df_labels_RL["run"] = "RL"
        df_events = concat(objs=[df_labels_LR, df_labels_RL]).reset_index(drop=True)
        df_events["condition"] = lis_labels_LR + lis_labels_RL
        df_events["condition_coded"] = Categorical(df_events["condition"]).codes
        df_events = df_events.reindex(labels=["run", 'onset', 'duration', 'trial_type', 'condition', 'condition_coded'], axis="columns")

        # --- Saving Beta Files -------------------------------------------------------------------

        for int_idx_beta_file, nib_beta_file in enumerate(lis_subject_betas):

            arr_beta_data = nib_beta_file.get_fdata()
            int_label = df_events.loc[int_idx_beta_file, "condition_coded"]
            
            str_beta_filename = f"sub-{int_subject_id}_{int_idx_beta_file}_{df_events.loc[int_idx_beta_file, "trial_type"].replace("_", "")}_betas"

            if str_subject in lis_subjects_missing_npz:

                arr_beta_masked = apply_mask(imgs=nib_beta_file, mask_img=img_mask)
                arr_beta_masked_reshaped = reshape(arr_beta_masked, shape=(1, len(arr_beta_masked)))
                int_label = int_label

                str_beta_filename_npz = str_beta_filename + ".npz"
                str_path_file = join(str_path_subject_betas, str_beta_filename_npz)

                savez_compressed(file=str_path_file, data=arr_beta_masked_reshaped, labels=int_label)

            if str_subject in lis_subjects_missing_pt:

                ten_beta = tensor(arr_beta_data, dtype=to_float32)
                lon_label = tensor(int_label, dtype=long)

                str_beta_filename_pt = str_beta_filename + ".pt"
                str_path_file = join(str_path_subject_betas, str_beta_filename_pt)

                save(obj={"data":ten_beta, "label": lon_label, "subject":int_subject_id}, f=str_path_file)

        print(f"    Beta files saved at {str_path_subject_betas}")

        if not isfile(join(str_data_path, "Events.csv")):
        
            df_events.to_csv(path_or_buf= join(str_data_path, "Events.csv"), index=False)

            print(f"    Event file saved at ...{str_data_path}")

        print("")