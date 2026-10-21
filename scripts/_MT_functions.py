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

# -------------------------------------------------------------------------------------------------

def get_runs():

    return ["LR", "RL"]

# -------------------------------------------------------------------------------------------------

def get_subjects(pat_data:str, flo_ratio_subjects:float=1)->list:

    if flo_ratio_subjects < 0.1 or flo_ratio_subjects > 1:

        raise ValueError("'flo_ratio_subjects must be >= 0.1 and <= 1")

    from numpy import round
    from numpy.random import shuffle
    from pathlib import Path


    lis_subjects = [subject for subject in pat_data.iterdir() if subject.stem.isdigit() or subject.name.startswith("sub-")]

    int_subjects_total = len(lis_subjects)
    int_subjects_part = int(round(flo_ratio_subjects * int_subjects_total))

    lis_subjects_new = lis_subjects[:int_subjects_part]

    #print(f"Subjects total: {f"{int_subjects_total:>3}"}")
    #print(f"Subjects_ratio: {f"{int((flo_ratio_subjects*100)):3}"}%")
    #print(f"Subjects new: {f"{int_subjects_part:>5}"}")
    #print("")

    return lis_subjects

# -------------------------------------------------------------------------------------------------

def get_tasks():

    return ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]

# -------------------------------------------------------------------------------------------------

def get_codes():

    dic_codes = {"neut":0, "fear":1, "loss_event":2, "win_event":3, "neut_event":4, "story":5, "math":6, "rh":7, "lf":8, "t":9,
                 "rf":10, "lh":11, "relation":12, "match":13, "mental":14, "rnd":15, "2bk_tools":16, "0bk_body":17, "2bk_faces":18,
                 "0bk_tools":19, "2bk_body":20, "2bk_places":21, "0bk_faces":22, "0bk_places":23}

    return dic_codes

# -------------------------------------------------------------------------------------------------

def compute_masks_intersect(pat_data:str, boo_force_new_files:bool=False):

    from nibabel import save as save_img
    from nilearn.image import load_img
    from nilearn.masking import intersect_masks
    from pathlib import Path

    from _MT_functions import get_subjects

    #pat_data = Path("/home/daniel/Desktop/Python/MasterThesis/data_raw")
    #boo_force_new_files = False

    lis_subjects = get_subjects(pat_data=pat_data)
    lis_tasks = get_tasks()
    lis_runs = get_runs()
    lis_task_masks_intersect = []
    pat_mask_all = pat_data/f"task-ALL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz"
    str_mask_all = pat_mask_all.name

    print(f"Path: {pat_data}")
    print("")

    for str_task in lis_tasks:

        str_mask_name = f"task-{str_task}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz"
        pat_mask_file = pat_data/str_mask_name

        if not pat_mask_file.is_file() or boo_force_new_files:

            print(f"{f"{str_mask_name:<88}"}\033[91mFILE MISSING\033[0m", end="\r")

            lis_task_masks = []

            for pat_subject in lis_subjects:

                str_subject = pat_subject.name

                for str_run in lis_runs:

                    img_subject_run_mask = load_img(pat_data/str_subject/"func"/f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz")

                    lis_task_masks.append(img_subject_run_mask)

            img_mask_intersect = intersect_masks(mask_imgs=lis_task_masks, threshold=1)

            lis_task_masks.append(img_mask_intersect)
            
            save_img(img=img_mask_intersect, filename=pat_mask_file)

            print(f"{f"{str_mask_name:<88}"}\033[92mfile created\033[0m")

        else:

            print(f"{f"{str_mask_name:<88}"}\033[92mfile found\033[0m")

    print("")

    if not pat_mask_all.is_file():

        print(f"{f"{str_mask_all:<88}"}\033[91mFILE MISSING\033[0m", end="\r")

        lis_task_masks_intersect = [mask for mask in pat_data.iterdir() if mask.name.endswith("intersect.nii.gz")]

        img_all_masks_intersect = intersect_masks(lis_task_masks_intersect, threshold=1)

        save_img(img=img_all_masks_intersect, filename=pat_mask_all)

        print(f"{f"{str_mask_all:<88}"}\033[92mfile created\033[0m")

    else:

        print(f"{f"{str_mask_all:<88}"}\033[92mfile found\033[0m")

    print("")
    print("-"*100)
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

# %%

def get_longest_file_name(lis_files:list)->tuple[str, int]:

    str_longest_file_name = ""

    for str_file in lis_files:

        if len(str_file) > len(str_longest_file_name):

            str_longest_file_name = str_file

    int_len_longest_file_name = len(str_longest_file_name)

    return str_longest_file_name, int_len_longest_file_name

# %%