
def get_arguments():

    from argparse import ArgumentParser
    from pathlib import Path

    ap = ArgumentParser(description="???")

    ap.add_argument("--pat_data", required=False, default=Path(__file__).parent.parent/"data", help="???")
    ap.add_argument("--str_data_type", required=False, default="bold", help="???")
    ap.add_argument("--lis_tasks_train", required=False, default=["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL"], nargs="+", help="???")
    ap.add_argument("--lis_tasks_test", required=False, default=["EMOTION"], nargs="+", help="???")
    ap.add_argument("--int_n_subjects_train", required=False, default=2, type=int, help="???")
    ap.add_argument("--int_n_subjects_test", required=False, default=1, type=int, help="???")
    ap.add_argument("--int_epochs", required=False, default=3, type=int, help="???")
    ap.add_argument("--int_batch_size_train", required=False, default=16, type=int, help="???")
    ap.add_argument("--int_batch_size_test", required=False, default=1, type=int, help="???")
    ap.add_argument("--int_progress_bar_size", required=False, default=4, type=int, help="???")
    ap.add_argument("--int_seed", required=False, default=42, type=int, help="???")
    ap.add_argument("--int_n_jobs", required=False, default=-1, type=int, help="???")
    ap.add_argument("--flo_learning_rate", required=False, default=0.001, help="CNN ONLY")
    ap.add_argument("--flo_dropout", required=False, default=0.2, help="CNN ONLY")

    return ap.parse_args()

# -------------------------------------------------------------------------------------------------

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

def get_settings_summary(args, int_n_subjects_total, lis_classes_train, pat_results):

    dic_codes = get_codes()
    dic_conditions = get_conditions()

    print("")
    print(f"\033[1mPaths\033[0m")
    print(f"Subjects: {args.pat_data}")
    print(f"Results: {pat_results}")  
    print("")
    print("\033[1mSubjects\33[0m")
    print(f"Total: {int_n_subjects_total}")
    print(f"Training: {args.int_n_subjects_train}")
    print(f"Testing: {args.int_n_subjects_test}")
    print("")
    print("\033[1mTasks:\033[0m")
    print(f"Training: {args.lis_tasks_train}")
    print(f"Testing: {args.lis_tasks_test}")
    print(f"Categories: {len(lis_classes_train)}")
    print(f"Chance Level: {100*(1/len(lis_classes_train))}%")
    print("")
    print("\33[1mParameters:\033[0m")
    print(f"Epochs: {args.int_epochs}")
    print(f"Batch size training: {args.int_batch_size_train}")
    print(f"Batch size testing: {args.int_batch_size_test}")
    print(f"N_jobs: {args.int_n_jobs}")
    print(f"Random state: {args.int_seed}")
    print("")

# -------------------------------------------------------------------------------------------------

def get_subjects(pat_data:str)->list:

    return  [subject for subject in pat_data.iterdir() if subject.stem.isdigit() or subject.name.startswith("sub-")]

# -------------------------------------------------------------------------------------------------

def get_tasks():

    return ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]

# -------------------------------------------------------------------------------------------------

def get_codes():

    dic_codes = {"neut":0, "fear":1, "loss_event":2, "win_event":3, "neut_event":4, "story":5, "math":6, "rh":7, "lf":8, "t":9,
                 "rf":10, "lh":11, "relation":12, "match":13, "mental":14, "rnd":15, "tools":16, "body":17, "faces":18, "places":20}

    return dic_codes

# -------------------------------------------------------------------------------------------------

def get_conditions():

    dic_conditions = {"EMOTION": ["neut", "fear"], "GAMBLING": ["loss_event", "win_event", "neut_event"], "LANGUAGE": ["story", "math"], 
                     "MOTOR": ["rh", "lf", "t", "rf", "lh"], "RELATIONAL": ["relation", "match"], "SOCIAL": ["mental", "rnd"], "WM": ["tools", "body", "faces", "places"]}

    return dic_conditions

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

# %% ----------------------------------------------------------------------------------------------

def get_longest_file_name(lis_files:list)->tuple[str, int]:

    str_longest_file_name = ""

    for str_file in lis_files:

        if len(str_file) > len(str_longest_file_name):

            str_longest_file_name = str_file

    int_len_longest_file_name = len(str_longest_file_name)

    return str_longest_file_name, int_len_longest_file_name

# %% ----------------------------------------------------------------------------------------------

def get_classes_training():

    args = get_arguments()
    dic_conditions = get_conditions()
    dic_codes = get_codes()

    lis_classes_train = []

    for str_key in args.lis_tasks_train:

        lis_classes_train = lis_classes_train + dic_conditions[str_key]

    lis_classes_coded_train = [dic_codes[x] for x in lis_classes_train]

    return lis_classes_train, lis_classes_coded_train

# %% ----------------------------------------------------------------------------------------------

def get_split_files():

    args = get_arguments()
    lis_subjects = get_subjects(args.pat_data)

    from sklearn.model_selection import train_test_split

    lis_subjects_train, lis_subjects_test = train_test_split(lis_subjects, train_size=args.int_n_subjects_train, test_size=args.int_n_subjects_test, random_state=args.int_seed)

    lis_files_train = []
    lis_files_test = []

    from regex import search

    for pat_subject in lis_subjects:

        if args.str_data_type == "bold":

            lis_subject_files = [x for x in (pat_subject/"func"/"bold_files").iterdir() if x.name.endswith("bold.npz")]

            if pat_subject in lis_subjects_train:

                for str_task in args.lis_tasks_train:

                    lis_matches = [pat_file for pat_file in lis_subject_files if search(pattern=str_task, string=pat_file.name)]

                    lis_files_train += lis_matches

            if pat_subject in lis_subjects_test:

                for str_task in args.lis_tasks_test:

                    lis_matches = [pat_file for pat_file in lis_subject_files if search(pattern=str_task, string=pat_file.name)]

                    lis_files_test += lis_matches

    return lis_files_train, lis_files_test, lis_subjects_train, lis_subjects_test