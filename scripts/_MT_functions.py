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

def get_mask_intersect(data_path:str, force_new_file:bool=False):

    from nibabel import save as save_img
    from nilearn.image import load_img
    from nilearn.masking import intersect_masks
    import os

    from scripts._MT_functions import get_subjects

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

    from scripts._MT_functions import get_mask_intersect, get_subjects

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

    from scripts._MT_functions import get_subjects

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
    