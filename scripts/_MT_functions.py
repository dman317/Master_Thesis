def get_hardware():

    # --- CPU -------------------------------------------------------------------------------------

    from platform import machine

    try:

        machine()

        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    str_cpu = line.strip()
                    break

    except:

        str_cpu = "unknown"

    # --- Mainboard -------------------------------------------------------------------------------

    try:

        with open(file="/sys/class/dmi/id/board_vendor") as x:

            str_mainboard_vendor = x.read().strip()

    except:

        str_mainboard_vendor = ""

    try:

        with open(file="/sys/class/dmi/id/board_name") as x:

            str_mainboard_model = x.read().strip()

    except:

        str_mainboard_model = ""

    try:

        with open(file="/sys/class/dmi/id/board_version") as x:

            str_mainboard_version = x.read().strip()

    except:

        str_mainboard_version = ""

    # --- RAM -------------------------------------------------------------------------------------

    

    return str_mainboard_vendor + ";" + str_mainboard_model + ";" + str_mainboard_version

# %%

def get_subjects(data_path:str, flo_ratio_subjects:float=1, boo_verbose:bool=False)->list:

    if flo_ratio_subjects < 0.1 or flo_ratio_subjects > 1:

        raise ValueError("'flo_ratio_subjects must be >= 0.1 and <= 1")

    from numpy import round
    from numpy.random import shuffle
    from os import listdir

    #data_path = "/home/daniel/Desktop/Python/Data/MRI/HCP_AWS" 
    #boo_verbose = True
    #flo_ratio_subjects = .8

    lis_subjects_full = [x for x in listdir(data_path) if x.startswith("sub-")]

    int_subjects_total = len(lis_subjects_full)
    int_subjects_part = int(round(flo_ratio_subjects * int_subjects_total))

    lis_subjects_new = lis_subjects_full[:int_subjects_part]

    if boo_verbose:

        print(f"Subjects total: {f"{int_subjects_total:>3}"}")
        print(f"Subjects_ratio: {f"{int((flo_ratio_subjects*100)):3}"}%")
        print(f"Subjects new: {f"{int_subjects_part:>5}"}")
        print("")

    return lis_subjects_new

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def compute_betas(data_path:str, force_new_files:bool=False, boo_verbose:bool=False):

    from matplotlib.pyplot import subplots
    from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
    from nilearn.image import load_img
    from nilearn.masking import apply_mask
    from nilearn.plotting import plot_design_matrix
    from numpy import arange, float16 as np_float32, int32 as np_int32, round, savez_compressed, unique, reshape
    from os import listdir, mkdir
    from os.path import isdir, isfile, join
    from pandas import Categorical, concat, DataFrame, read_csv
    from torch import float32 as to_float32, long, save, tensor

    from _MT_functions import get_subjects, get_mask_intersect

    lis_subjects = get_subjects(data_path=data_path)

    img_mask = get_mask_intersect(data_path=data_path)

    # --- Checking Beta Directories & Files ---------------------------------------------------------------

    print("Handling Beta Directories and Files...")
    print("")

    lis_subjects_missing_npz = []
    lis_subjects_missing_pt = []
    
    for int_subject_i, str_subject in enumerate(lis_subjects, start=1):

        if boo_verbose:

            print(f"{str_subject} ({int_subject_i}/{len(lis_subjects)})")
            print("    checking for beta directory...", end="\r")
            
        str_path_subject_betas = join(data_path, str_subject, "func", "betas")

        if not isdir(str_path_subject_betas):

            if boo_verbose:
            
                print("    checking for beta directory...no directory found")

            mkdir(str_path_subject_betas)

            if boo_verbose:

                print(f"    New directory created at: {str_path_subject_betas}")

            lis_subjects_missing_npz.append(str_subject)
            lis_subjects_missing_pt.append(str_subject)

            if boo_verbose:

                print(f"    added '{str_subject}' to list of subjects with missing or incomplete betas")

        else:

            if boo_verbose:

                print("    checking beta directory...directory found")
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

                if boo_verbose:
                    
                    print("    checking for beta files...files missing or incomplete")
                    print(f"    added '{str_subject}' to list of subjects with missing or incomplete betas")

            else:

                if boo_verbose:

                    print("    checking for beta files...files found")

    if boo_verbose:

        print("")

    print("-"*100)
    print("")

    # --- Computation of Beta Files ---------------------------------------------------------------
    
    if bool(lis_subjects_missing_npz) or bool(lis_subjects_missing_pt):

        print("")
        str_temp = "--- Computation of Beta Files "
        print(str_temp + "-"*(100-len(str_temp)))
        print("")

        arr_subjects_with_incomplete_data = unique(lis_subjects_missing_npz + lis_subjects_missing_pt)
        int_n_subjects_with_incomplete_data = len(arr_subjects_with_incomplete_data)

        for int_i_subject, str_subject in enumerate(arr_subjects_with_incomplete_data, start=1):

            int_subject_id = str_subject.split("-")[1]
            str_path_subject_betas = join(data_path, str_subject, "func", "betas")

            print(f"{str_subject} ({int_i_subject}/{int_n_subjects_with_incomplete_data})")

            str_subject_func_path = join(data_path, str_subject, "func/")

            img_func_LR = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"))
            img_mask_LR = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz"))
            df_events_LR = read_csv(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-LR_desc-EV_summary.csv"), sep=",")

            img_func_RL = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"))
            img_mask_RL = load_img(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz"))
            df_events_RL = read_csv(join(str_subject_func_path, f"{str_subject}_task-EMOTION_run-RL_desc-EV_summary.csv"), sep=",")

            # --- Transform Data ----------------

            int_n_scans_LR = img_func_LR.header.get_data_shape()[3]                         # extract the total number of datapoints (scans) --> 172
            flo_t_r_LR = float(img_func_LR.header.get_zooms()[3])                          # extract the time intervall between each datapoinr --> .72
            lis_frame_times_LR = arange(int_n_scans_LR) * flo_t_r_LR                       # calculate the time (s) when each datapoint was taken

            int_n_scans_RL = img_func_RL.header.get_data_shape()[3]
            flo_t_r_RL = float(img_func_RL.header.get_zooms()[3])
            lis_frame_times_RL = arange(int_n_scans_RL) * flo_t_r_RL

            # --- Setup Unique Block Conditions ---------------------------

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

            # --- GLM ---------------------------

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

            # --- Saving Beta Files -------------

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

            if not isfile(join(data_path, "Events.csv")):
            
                df_events.to_csv(path_or_buf= join(data_path, "Events.csv"), index=False)

                print(f"    Event file saved at ...{data_path}")

            print("")

        print("-" * 100)
        print("")

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def get_mask_intersect(data_path:str, force_new_file:bool=False):

    from nibabel import save as save_img
    from nilearn.image import load_img
    from nilearn.masking import intersect_masks
    import os

    from _MT_functions import get_subjects

    lis_subjects = get_subjects(data_path=data_path)
    lis_runs = ["LR", "RL"]
    str_mask_name = "brain_mask_intersect.nii.gz"
    str_mask_path = os.path.join(data_path, str_mask_name)

    if not os.path.isfile(str_mask_path) or force_new_file:

        print(f"Could not find mask!")
        print("Computing new mask...", end="\r")

        lis_subjects_masks = []

        for str_subject in lis_subjects:

            for str_run in lis_runs:

                img_subject_run_mask = load_img(f"{data_path}/{str_subject}/func/{str_subject}_task-EMOTION_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz")

                lis_subjects_masks.append(img_subject_run_mask)

        img_mask_intersect = intersect_masks(mask_imgs=lis_subjects_masks, threshold=1)

        print("Computing new mask...done!")
        
        print(f"Saving mask...", end="\r")

        save_img(img=img_mask_intersect, filename=os.path.join(data_path, str_mask_name))

        print(f"Saving mask...done!")
        print("")

    return load_img(str_mask_path)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def compute_bold_files(data_path:str, boo_apply_mask:bool=True, boo_standardize:bool=True, verbose:bool=False, force_new_files:bool=False):

    from numpy import arange, float16, savez_compressed, round, where
    from os.path import isfile, join
    import pandas as pd
    from nilearn.image import load_img, index_img
    from nilearn.masking import apply_mask
    from sklearn.preprocessing import StandardScaler

    from _MT_functions import get_subjects, get_mask_intersect

    # --- Handle Basic Data ------------------

    lis_subjects = get_subjects(data_path=data_path)
    lis_runs = ["LR", "RL"]

    print("Checking for missing files...")
    print("")

    # --- Mask ----------------------------------

    # returns a mask created by the intersection of all subjects masks -> contains only voxels that can be found in ALL subject masks
    img_mask_intersect = get_mask_intersect(data_path=data_path)

    # --- Handle .npz files ------------------

    df_events = pd.read_csv("/home/daniel/Desktop/Python/Data/MRI/HCP_AWS/decoding.csv", sep=",").sort_values("tr")

    for str_subject in lis_subjects:

        str_subject_path = join(data_path, str_subject, "func/")

        for run in lis_runs:

            str_file_name = f"{str_subject}_task-EMOTION_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.npz"
            
            if not isfile(join(data_path, str_subject, "func", str_file_name)) or force_new_files:

                print(f"File '{str_file_name}' not found!")
                print("Commencing computation...", end="\r")

                img_func = load_img(join(data_path, str_subject, "func", f"{str_subject}_task-EMOTION_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"))

                int_n_scans = img_func.header.get_data_shape()[3]
                flo_t_r = img_func.header.get_zooms()[3]
                arr_scan_times = arange(int_n_scans) * flo_t_r

                arr_scan_times_rounded = round(a=arr_scan_times, decimals=2)
                arr_event_times_rounded = round(a=df_events["tr"].values, decimals=2)

                arr_match_indicies = []

                for event_time in arr_event_times_rounded:

                    match = where(event_time == arr_scan_times_rounded)[0]
                    
                    arr_match_indicies.append(match[0])
                
                img_func_filtered = index_img(imgs=img_func, index=arr_match_indicies)

                arr_func_filtered_masked = apply_mask(imgs=img_func_filtered, mask_img=img_mask_intersect).astype(float16)

                if boo_standardize:

                    scaler = StandardScaler()
                    arr_func_filtered_masked_scaled = scaler.fit_transform(arr_func_filtered_masked)

                    scaler = StandardScaler()
                    arr_func_filtered_scaled = scaler.fit_transform(img_func_filtered.get_fdata())

                arr_labels = df_events["true_state"].values

                str_file_path = join(str_subject_path, str_file_name)
            
                print("Commencing computation...done!")

                savez_compressed(file=str_file_path, data_masked=arr_func_filtered_masked_scaled, data_unmasked=arr_func_filtered_scaled, labels=arr_labels)

                print(f"File saved at {str_file_path}")
                print("")

        #TODO take care of the mask issue
        #TODO come up with a better name for the saved npz-file
        #TODO make sure labels are correctly assigned
        #TODO come up with better prints and make them coherrent across SVM and DL scripts

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def compute_synthetic_data(data_path:str, n_files:int=143,
                           flo_signal_strength:float=.6, flo_noise_low:float=.1, flo_noise_high:float=.2,
                           boo_standardize:bool=True, force_new_files:bool=False):

    from nibabel import Nifti1Image
    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.image import concat_imgs, load_img, math_img
    from nilearn.masking import apply_mask
    from numpy import savez_compressed
    from numpy.random import uniform
    from os.path import isfile, join
    from sklearn.preprocessing import StandardScaler

    from _MT_functions import get_mask_intersect, get_subjects

    lis_runs = ["LR", "RL"]

    lis_subjects = get_subjects(data_path=data_path)
    img_mask = get_mask_intersect(data_path=data_path)

    # --- Handle the Masks for Condition 1&2

    dic_atlas = fetch_atlas_harvard_oxford(data_dir="/home/daniel/Desktop/Python/Data/MRI/atlases", atlas_name="cort-maxprob-thr25-2mm")
    img_atlas = load_img(dic_atlas["maps"])
    df_atlas_labels = dic_atlas["lut"]

    dic_masks = {}

    for i_row, row in df_atlas_labels.iterrows():

        dic_masks[row["name"]] = math_img(formula=f"img01 == {row["index"]}", img01 = img_atlas)

    img_mask_roi_cond01 = load_img(dic_masks["Frontal Pole"])
    arr_mask_roi_cond01_fdata = img_mask_roi_cond01.get_fdata()

    img_mask_roi_cond02 = load_img(dic_masks["Occipital Pole"])
    arr_mask_roi_cond02_fdata = img_mask_roi_cond02.get_fdata()

    # --- Check if File Already Exists ------------------

    boo_files_missing = False
    lis_missing_files = []

    print("Checking for missing files...")
    print("")

    for str_subject in lis_subjects:

        str_subject_path = join(data_path, str_subject, "func")

        for str_run in lis_runs:

            str_file_name = f"{str_subject}_run-{str_run}_space-MNI152_noise-low-{flo_noise_low}_noise-high-{flo_noise_high}_signal-{flo_signal_strength}_synth.npz"
            str_file_path = join(str_subject_path, str_file_name)

            if not isfile(str_file_path) or force_new_files:

                # --- Compute File --------------

                boo_files_missing = True
                lis_missing_files.append(str_file_path)

                print(f"File '{str_file_name}' not found!")
                print("Commencing computation...", end="\r")

                lis_imgs = []
                lis_labels = []

                for i_file in range(int(n_files/2)):  #TODO make sure len is same as real data (total number)

                    arr_volume_cond01 = uniform(low=flo_noise_low, high=flo_noise_high, size=img_mask.shape)
                    arr_volume_cond01[arr_mask_roi_cond01_fdata > 0] += flo_signal_strength
                    img_cond01 = Nifti1Image(dataobj=arr_volume_cond01, affine=img_mask.affine)

                    lis_imgs.append(img_cond01)
                    lis_labels.append(0)

                    arr_volume_cond02 = uniform(low=flo_noise_low, high=flo_noise_high, size=img_mask.shape)
                    arr_volume_cond02[arr_mask_roi_cond02_fdata > 0] += flo_signal_strength
                    img_cond02 = Nifti1Image(dataobj=arr_volume_cond02, affine=img_mask.affine)

                    lis_imgs.append(img_cond02)
                    lis_labels.append(1)
                
                img_data = concat_imgs(lis_imgs)
                img_data_masked = apply_mask(imgs=img_data, mask_img=img_mask)

                if boo_standardize:

                    scaler = StandardScaler()
                    img_data_masked = scaler.fit_transform(img_data_masked)
                
                print("Commencing computation...done!")

                savez_compressed(file=str_file_path, data=img_data_masked, labels=lis_labels)

                print(f"Saved file at {str_file_path}")
                print("")
    
    if not boo_files_missing:

        print("All files have been found!")
        print("")

    else:

        print(f"{len(lis_missing_files)} files have been computed.")
        print("All files have been found!")
        print("")


# %% ----------------------------------------------------------------------------------------------

def remove_data_files(str_data_path:str, str_which_type:str, boo_remove:bool=True):

    #TODO add confirmation via input
    
    # --- Check Input ---------------------------

    #str_data_path = "/home/daniel/Desktop/Python/Data/MRI/HCP_AWS"   # just for testing & development
    #str_which_type = "betas-npz"                                     # just for testing & development

    lis_accepted_types = ["all", "bold", "bold-hcp", "bold-synth", "betas", "betas-npz", "betas-pt"]

    if str_which_type not in lis_accepted_types:

        raise ValueError(f"'str_which' must be one of the follwing: {lis_accepted_types}")

    # --- Load Modules --------------------------

    from os import listdir, remove
    from os.path import join

    from _MT_functions import get_subjects

    # --- Handle Basic Variables ----------------

    lis_subjects = get_subjects(data_path=str_data_path)
    lis_files_removed_bold = []
    lis_files_removed_betas = []
    
    # --- Handle Bold-Files ---------------------

    if str_which_type in ["all", "bold", "bold-hcp", "bold-synth"]:
        
        dic_types_bold = {"all":".npz", "bold":".npz", "bold-hcp":"bold.npz", "bold-synth":"synth.npz"}

        for subject in lis_subjects:

            str_subject_path_bold = join(str_data_path, subject, "func")

            for file in listdir(str_subject_path_bold):

                if file.endswith(dic_types_bold.get(str_which_type)):

                    str_file_path_bold = join(str_subject_path_bold, file)
                    lis_files_removed_bold.append(str_file_path_bold)

                    if boo_remove:

                        remove(str_file_path_bold)

    # --- Handle Beta-Files ---------------------

    if str_which_type in ["all", "betas", "betas-npz", "betas-pt"]:
        
        dic_types_betas = {"all":(".npz", ".pt"), "betas":(".npz", ".pt"), "betas-npz":"betas.npz", "betas-pt":"betas.pt"}

        for subject in lis_subjects:

            str_subject_path_betas = join(str_data_path, subject, "func", "betas")

            for file in listdir(str_subject_path_betas):

                if file.endswith(dic_types_betas.get(str_which_type)):

                    str_file_path_betas = join(str_subject_path_betas, file)
                    lis_files_removed_betas.append(str_file_path_betas)

                    if boo_remove:
                        
                        remove(str_file_path_betas)

    # --- Report --------------------------------

    print("Deleting files...")
    print("")
    print(f"Subjects: {len(lis_subjects)}")
    print(f"Selected type: '{str_which_type}'")
    print(f"Bold files removed: {len(lis_files_removed_bold)}")
    print(f"Beta files removed: {len(lis_files_removed_betas)}")
    print("")

# %%

def remove_log_files(str_data_path:str, boo_remove_files:bool=False):

    from os import listdir, remove
    from os.path import join

    str_logs_path = join(str_data_path, "logs")

    lis_log_files = [x for x in listdir(str_logs_path) if x.endswith(".csv")]
    int_n_log_files = len(lis_log_files)

    if boo_remove_files:

        for file in lis_log_files:

            str_log_file_path = join(str_logs_path, file)

            remove(str_log_file_path)

    print(f"Log files found: {int_n_log_files}")

    if boo_remove_files:

        print(f"Log files removed: {int_n_log_files}")

    else:

        print("Log files removed: 0")

    print("")


# %%

def get_memory(boo_verbose:bool = False):

    from numpy import round
    from psutil import virtual_memory

    vm = virtual_memory()

    flo_memory_used = round(vm.used/(1024**3), decimals=2)
    flo_memory_percent = round(vm.percent, decimals=2)
    flo_memory_available = round(vm.available/(1024**3), decimals=2)
    flo_memory_total = round(vm.total/(1024**3), decimals=2)

    int_format_width = 18

    if boo_verbose:

        print("")   
        print("Memory RAM")
        print(f"  {'Memory Used:':<{int_format_width}} {flo_memory_used} GB ({flo_memory_percent}%)")
        print(f"  {'Memory Available:':<{int_format_width}} {flo_memory_available} GB")
        print(f"  {'Memory Total:':<{int_format_width}} {flo_memory_total} GB")
        print("")

    return flo_memory_used, flo_memory_percent, flo_memory_available, flo_memory_used

# %%

def handle_logs(str_data_path, dic_log_epochs, dic_log_run, dic_log_subjects):

    from numpy import nan
    from os import mkdir
    from os.path import isdir, join
    from pandas import DataFrame

    str_logs_path = join(str_data_path, "logs")

    if not isdir(str_logs_path):

        mkdir(str_logs_path)

    # --- Log-Epochs ------------------------------------------------------------------------------

    lis_log_epochs_keys = list(dic_log_epochs.keys())
    df_log_epochs = DataFrame(index=lis_log_epochs_keys)

    for key in lis_log_epochs_keys:

        for sub_key in dic_log_epochs[key]:

            value = dic_log_epochs[key][sub_key]

            df_log_epochs.loc[key, sub_key] = value

    # --- Log-Run ---------------------------------------------------------------------------------

    lis_log_run_keys = list(dic_log_run.keys())
    df_log_run = DataFrame(index=lis_log_run_keys, columns=["String"])

    for key in lis_log_run_keys:

        df_log_run.loc[key, "String"] = dic_log_run[key]

    # --- Log-Subjects ----------------------------------------------------------------------------

    #TODO works but can be improved
    df_log_subjects = DataFrame(columns=["Subjects_Train", "Subjects_Test"])
    df_log_subjects["Subjects_Train"] = dic_log_subjects["Subjects_Train"]
    df_log_subjects["Subjects_Test"] = dic_log_subjects["Subjects_Test"] + [nan] * (len(dic_log_subjects["Subjects_Train"]) - len(dic_log_subjects["Subjects_Test"]))

    # --- Save Log-Files --------------------------------------------------------------------------
    
    int_run_id = df_log_run.loc["Run-ID", "String"]
    str_date = df_log_run.loc["Date", "String"]
    str_time = df_log_run.loc["Time", "String"]

    str_log_epochs_file_name = f"{int_run_id}_{str_date}_{str_time}_epochs.csv"
    df_log_epochs.to_csv(path_or_buf=join(str_logs_path, str_log_epochs_file_name))
    print(f"'{str_log_epochs_file_name}' saved at {str_logs_path}")

    str_log_run_file_name = f"{int_run_id}_{str_date}_{str_time}_run.csv"
    df_log_run.to_csv(path_or_buf=join(str_logs_path, str_log_run_file_name))
    print(f"'{str_log_run_file_name}' saved at {str_logs_path}")

    str_log_subjects_file_name = f"{int_run_id}_{str_date}_{str_time}_subjects.csv"
    df_log_subjects.to_csv(path_or_buf=join(str_logs_path, str_log_subjects_file_name), index=False)
    print(f"'{str_log_subjects_file_name}' saved at {str_logs_path}")

# %%

def progress_bar(int_n_train_batch, int_progress_bar_size, int_n_train_batches, int_format_width):

    from numpy import round

    int_batch_batches_ratio = int(round((int_n_train_batch * int_progress_bar_size)/int_n_train_batches, decimals=1)*10)
    str_train_progress_bar = "[" + "▌" * int_batch_batches_ratio  + "." * ((int_progress_bar_size*10) - int_batch_batches_ratio) + "]"

    return str_train_progress_bar
    